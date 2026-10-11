"""Explicit, single-device training for EEG transformer classification.

Constructing a trainer creates an optimizer and scheduler, but never trains,
writes a checkpoint, or seeds global random generators. Only ``fit`` performs
parameter updates. Seed the model's initialization separately before creating it.
Checkpoints resume at complete epoch boundaries. Exact data-order reproduction
requires main-process loaders (``num_workers=0``); custom dataset/sampler state
and worker-local random state remain the caller's responsibility.
Random-state restoration does not promise identical results across devices or
PyTorch versions.
"""

import inspect
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings; loaders are supplied by the caller.

    ``batch_size`` describes the expected training loader's batch size. Warmup
    and cosine decay advance once per completed epoch. ``output_dir`` must be
    explicit before ``fit`` can write ``best.pt`` and ``last.pt``. ``seed`` is
    applied on the first fit, after model initialization, and is not reapplied
    when resuming a checkpoint whose training has already started.
    """

    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 5
    clip_grad_norm: Optional[float] = 1.0
    patience: Optional[int] = 20
    min_delta: float = 0.0
    device: str = "cpu"
    seed: int = 42
    output_dir: Optional[str] = None

    def __post_init__(self):
        for name in ("epochs", "batch_size", "warmup_epochs", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("{} must be an integer".format(name))
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not 0 <= self.warmup_epochs <= self.epochs:
            raise ValueError("warmup_epochs must lie between zero and epochs")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must lie in [0, 2**32)")
        if self.patience is not None and (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience < 1
        ):
            raise ValueError("patience must be positive or None")
        for name in ("lr", "weight_decay", "min_delta"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and non-negative".format(name))
        if self.lr == 0:
            raise ValueError("lr must be positive")
        if self.clip_grad_norm is not None and (
            not math.isfinite(self.clip_grad_norm) or self.clip_grad_norm <= 0
        ):
            raise ValueError("clip_grad_norm must be positive or None")
        if torch.device(self.device).type not in ("cpu", "cuda"):
            raise ValueError("Only explicit CPU or CUDA devices are supported")
        if self.output_dir is not None and not str(self.output_dir):
            raise ValueError("output_dir must be non-empty or None")

    def to_dict(self):
        """Return serializable settings without creating output directories."""
        result = asdict(self)
        if result["output_dir"] is not None:
            result["output_dir"] = str(result["output_dir"])
        return result

    @classmethod
    def from_dict(cls, values):
        """Reconstruct settings; unknown fields are rejected by the constructor."""
        return cls(**values)


class TransformerTrainer:
    """AdamW trainer for a model exposing ``forward`` and ``get_config``.

    Batches contain floating ``X`` of shape ``(B, C, T)``, integer ``y`` of
    shape ``(B,)``, and optionally mappings ``auxiliary`` and ``auxiliary_mask``.
    Masks use ``True`` for valid tokens. Features are moved to the model's
    floating dtype/device; labels and masks must already have long/bool dtype.
    ``forward_batch`` and ``compute_loss`` preserve the current model mode and
    gradient context. For prediction without training-buffer changes, use
    ``evaluate`` or explicitly select ``model.eval()`` and ``torch.no_grad()``.
    The default criterion is cross-entropy. A custom scalar classification loss
    must be an ``nn.Module`` with serializable ``get_config()`` and no trainable
    parameters; reconstruct it explicitly before loading its checkpoint state.
    Cross-entropy must use mean reduction; weighted cross-entropy is aggregated
    by target-weight totals. Custom losses must return a mean per trial and are
    aggregated by sample count. Accuracy measures argmax against every label;
    ignored targets are not supported by this classification recipe.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig, criterion=None):
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("The requested CUDA device is unavailable")
        if not callable(getattr(model, "get_config", None)):
            raise TypeError("model must expose get_config() for strict checkpoints")
        self.model_config = json.loads(json.dumps(model.get_config(), sort_keys=True))
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        if not isinstance(self.criterion, nn.Module):
            raise TypeError("criterion must be an nn.Module returning a scalar loss")
        if (
            isinstance(self.criterion, nn.CrossEntropyLoss)
            and self.criterion.reduction != "mean"
        ):
            raise ValueError("CrossEntropyLoss must use reduction='mean'")
        if any(parameter.requires_grad for parameter in self.criterion.parameters()):
            raise ValueError(
                "Put trainable loss parameters in the model, not criterion"
            )
        self.criterion.to(device=self.device, dtype=next(model.parameters()).dtype)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            weight = self.criterion.weight
            if weight is not None and (
                weight.ndim != 1
                or weight.numel() == 0
                or not torch.isfinite(weight).all().item()
                or (weight < 0).any().item()
                or float(weight.sum()) <= 0
            ):
                raise ValueError("Class weights must be finite non-negative values")
        self.criterion_config = self._criterion_identity()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, self._lr_multiplier
        )
        self.epoch = 0
        self.history = []
        self.best_loss = math.inf
        self.bad_epochs = 0
        self._seeded = False
        self._generators = {}
        self._generator_states = {}

    def _lr_multiplier(self, epoch):
        warmup = self.config.warmup_epochs
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, (epoch - warmup) / max(1, self.config.epochs - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _prepare_batch(self, batch):
        if not isinstance(batch, Mapping) or not {"X", "y"}.issubset(batch):
            raise ValueError("batch must contain X and y")
        X, y = batch["X"], batch["y"]
        if not torch.is_tensor(X) or X.ndim != 3 or not X.is_floating_point():
            raise ValueError("X must be a floating tensor of shape (B, C, T)")
        if (
            not torch.is_tensor(y)
            or y.dtype != torch.long
            or y.ndim != 1
            or len(y) != len(X)
            or len(y) == 0
        ):
            raise ValueError("y must be a non-empty long tensor of shape (B,)")
        dtype = next(self.model.parameters()).dtype
        auxiliary = batch.get("auxiliary")
        masks = batch.get("auxiliary_mask")
        if auxiliary is not None:
            if not isinstance(auxiliary, Mapping):
                raise ValueError("auxiliary must be a mapping")
            converted = {}
            for name, value in auxiliary.items():
                if not torch.is_tensor(value) or not value.is_floating_point():
                    raise ValueError("auxiliary features must be floating tensors")
                converted[name] = value.to(device=self.device, dtype=dtype)
            auxiliary = converted
        if masks is not None:
            if not isinstance(masks, Mapping):
                raise ValueError("auxiliary_mask must be a mapping")
            converted = {}
            for name, value in masks.items():
                if not torch.is_tensor(value) or value.dtype != torch.bool:
                    raise ValueError("auxiliary masks must be bool tensors")
                converted[name] = value.to(self.device)
            masks = converted
        return (
            X.to(device=self.device, dtype=dtype),
            y.to(self.device),
            auxiliary,
            masks,
        )

    def forward_batch(self, batch):
        """Compute logits without an optimizer update; preserve model mode."""
        X, _, auxiliary, masks = self._prepare_batch(batch)
        return self.model(X, auxiliary=auxiliary, auxiliary_mask=masks)

    def _loss(self, logits, y):
        if (
            not torch.is_tensor(logits)
            or not logits.is_floating_point()
            or logits.ndim != 2
            or len(logits) != len(y)
            or logits.shape[1] == 0
        ):
            raise ValueError("Model logits must have shape (B, n_classes)")
        if not torch.isfinite(logits).all().item():
            raise ValueError("Model logits must be finite")
        if (y < 0).any().item() or (y >= logits.shape[1]).any().item():
            raise ValueError("Targets must lie in [0, n_classes)")
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            if (y == self.criterion.ignore_index).any().item():
                raise ValueError("Ignored targets are not supported")
            weight = self.criterion.weight
            if weight is not None and (
                weight.ndim != 1
                or len(weight) != logits.shape[1]
                or not torch.isfinite(weight).all().item()
                or (weight < 0).any().item()
            ):
                raise ValueError("Class weights must be finite non-negative values")
            if self._loss_denominator(y) <= 0:
                raise ValueError("The target-weight total must be positive")
        loss = self.criterion(logits, y)
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("criterion must return a scalar tensor loss")
        if not torch.isfinite(loss).item():
            raise ValueError("The classification loss is non-finite")
        return loss

    def _loss_denominator(self, y):
        if (
            isinstance(self.criterion, nn.CrossEntropyLoss)
            and self.criterion.weight is not None
        ):
            return float(self.criterion.weight[y].sum())
        return float(len(y))

    def compute_loss(self, batch):
        """Return a scalar loss with its graph; never update parameters."""
        X, y, auxiliary, masks = self._prepare_batch(batch)
        logits = self.model(X, auxiliary=auxiliary, auxiliary_mask=masks)
        return self._loss(logits, y)

    def _run_epoch(self, loader, training):
        loss_sum, correct, samples, grad_sum, batches = 0.0, 0, 0, 0.0, 0
        loss_denominator = 0.0
        self.model.train(training)
        with torch.set_grad_enabled(training):
            for batch in loader:
                X, y, auxiliary, masks = self._prepare_batch(batch)
                if training:
                    self.optimizer.zero_grad()
                logits = self.model(X, auxiliary=auxiliary, auxiliary_mask=masks)
                loss = self._loss(logits, y)
                if training:
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad_norm or math.inf
                    )
                    if not math.isfinite(float(norm)):
                        raise ValueError("The gradient norm is non-finite")
                    grad_sum += float(norm)
                    self.optimizer.step()
                denominator = self._loss_denominator(y)
                loss_sum += float(loss.detach()) * denominator
                loss_denominator += denominator
                correct += int((logits.detach().argmax(dim=1) == y).sum())
                samples += len(y)
                batches += 1
        if samples == 0:
            raise ValueError("A training or validation loader is empty")
        return {
            "loss": loss_sum / loss_denominator,
            "accuracy": correct / samples,
            "samples": samples,
            "grad_norm": grad_sum / batches if training else None,
        }

    def evaluate(self, loader):
        """Evaluate under no-grad and restore the preceding model mode."""
        training = self.model.training
        try:
            result = self._run_epoch(loader, training=False)
            return {key: result[key] for key in ("loss", "accuracy", "samples")}
        finally:
            self.model.train(training)

    def _bind_generators(self, train_loader, valid_loader):
        self._generators = {}
        for name, loader in (("train", train_loader), ("valid", valid_loader)):
            for part, obj in (
                ("loader", loader),
                ("sampler", getattr(loader, "sampler", None)),
            ):
                generator = getattr(obj, "generator", None)
                if isinstance(generator, torch.Generator):
                    self._generators[name + "." + part] = generator
        for name, state in self._generator_states.items():
            if name not in self._generators:
                raise ValueError("Missing checkpoint loader generator: " + name)
            self._generators[name].set_state(state)
        self._generator_states = {}

    def fit(self, train_loader, valid_loader):
        """Explicitly train, validate, and save best/last epoch checkpoints.

        Validation loss controls early stopping, using ``min_delta``. The model
        remains at the last completed epoch; load ``best.pt`` explicitly for
        best-epoch inference. A failed epoch is not recorded as completed.
        """
        if self.config.output_dir is None:
            raise ValueError("Set an explicit output_dir before calling fit")
        batch_size = getattr(train_loader, "batch_size", None)
        if batch_size is not None and batch_size != self.config.batch_size:
            raise ValueError("Training loader batch_size differs from the config")
        if self.config.patience is not None and self.bad_epochs >= self.config.patience:
            raise ValueError("This checkpoint has already reached early stopping")
        self._bind_generators(train_loader, valid_loader)
        if not self._seeded:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.seed)
            self._seeded = True
        output_dir = Path(self.config.output_dir)
        while self.epoch < self.config.epochs:
            lr = self.optimizer.param_groups[0]["lr"]
            train = self._run_epoch(train_loader, training=True)
            valid = self.evaluate(valid_loader)
            improved = valid["loss"] < self.best_loss - self.config.min_delta
            if improved:
                self.best_loss, self.bad_epochs = valid["loss"], 0
            else:
                self.bad_epochs += 1
            self.epoch += 1
            self.history.append(
                {
                    "epoch": self.epoch,
                    "lr": lr,
                    "train_loss": train["loss"],
                    "valid_loss": valid["loss"],
                    "train_accuracy": train["accuracy"],
                    "valid_accuracy": valid["accuracy"],
                    "grad_norm": train["grad_norm"],
                }
            )
            self.scheduler.step()
            if improved:
                self.save_checkpoint(output_dir / "best.pt")
            self.save_checkpoint(output_dir / "last.pt")
            if (
                self.config.patience is not None
                and self.bad_epochs >= self.config.patience
            ):
                break
        return deepcopy(self.history)

    def _model_identity(self):
        cls = type(self.model)
        return cls.__module__ + "." + cls.__qualname__

    def _criterion_identity(self):
        cls = type(self.criterion)
        if cls is nn.CrossEntropyLoss:
            settings = {
                name: getattr(self.criterion, name)
                for name in ("ignore_index", "reduction", "label_smoothing")
                if hasattr(self.criterion, name)
            }
            settings["weight"] = (
                None
                if self.criterion.weight is None
                else self.criterion.weight.detach().cpu().tolist()
            )
        elif callable(getattr(self.criterion, "get_config", None)):
            settings = self.criterion.get_config()
        else:
            raise TypeError("A custom criterion must expose serializable get_config()")
        return {
            "class": cls.__module__ + "." + cls.__qualname__,
            "config": json.loads(json.dumps(settings, sort_keys=True)),
        }

    def save_checkpoint(self, path):
        """Explicitly save an atomic local checkpoint, including an untrained one."""
        if self.model.get_config() != self.model_config:
            raise ValueError("Model configuration changed after trainer construction")
        if self._criterion_identity() != self.criterion_config:
            raise ValueError(
                "Criterion configuration changed after trainer construction"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
            ),
        }
        generators = dict(self._generator_states)
        generators.update(
            {name: gen.get_state() for name, gen in self._generators.items()}
        )
        payload = {
            "format_version": 1,
            "model_class": self._model_identity(),
            "model_config": self.model_config,
            "model": self.model.state_dict(),
            "criterion_config": self.criterion_config,
            "criterion": self.criterion.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self.epoch,
            "history": deepcopy(self.history),
            "best_loss": self.best_loss,
            "bad_epochs": self.bad_epochs,
            "seeded": self._seeded,
            "rng": rng,
            "loader_generators": generators,
        }
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def load_checkpoint(self, path, *, trusted=False):
        """Restore state without training; only load trusted local checkpoints.

        Optimizer/Python/NumPy state requires pickle deserialization. Set
        ``trusted=True`` only for checkpoints whose provenance you trust.
        PyTorch versions exposing ``weights_only`` receive an explicit False;
        older supported versions are called without that keyword. Device and
        output directory may change; architecture and other settings may not.
        Custom criteria must be reconstructed by the caller with matching
        ``get_config()`` settings; their tensor state is restored from the file.
        """
        if not trusted:
            raise ValueError("Only load trusted local checkpoints with trusted=True")
        kwargs = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        payload = torch.load(Path(path), **kwargs)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        if (
            payload["model_class"] != self._model_identity()
            or payload["model_config"] != self.model_config
        ):
            raise ValueError("Checkpoint model architecture/config does not match")
        if payload["criterion_config"] != self.criterion_config:
            raise ValueError("Checkpoint criterion settings do not match")
        saved_config = TrainingConfig.from_dict(payload["config"]).to_dict()
        current_config = self.config.to_dict()
        for name in ("device", "output_dir"):
            saved_config.pop(name)
            current_config.pop(name)
        if saved_config != current_config:
            raise ValueError("Checkpoint training settings do not match")
        current_state, saved_state = self.model.state_dict(), payload["model"]
        if current_state.keys() != saved_state.keys() or any(
            current_state[name].shape != saved_state[name].shape
            or current_state[name].dtype != saved_state[name].dtype
            for name in current_state
        ):
            raise ValueError("Checkpoint model state shapes/dtypes do not match")
        epoch = payload["epoch"]
        if not isinstance(epoch, int) or not 0 <= epoch <= self.config.epochs:
            raise ValueError("Invalid completed-epoch count")
        if len(payload["history"]) != epoch:
            raise ValueError("Checkpoint history does not match completed epochs")
        criterion_state = self.criterion.state_dict()
        if criterion_state.keys() != payload["criterion"].keys() or any(
            criterion_state[name].shape != payload["criterion"][name].shape
            or criterion_state[name].dtype != payload["criterion"][name].dtype
            for name in criterion_state
        ):
            raise ValueError("Checkpoint criterion state shapes/dtypes do not match")
        cuda_rng = payload["rng"]["cuda"]
        if self.device.type == "cuda" and cuda_rng:
            if len(cuda_rng) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA RNG device count does not match")
        self.model.load_state_dict(saved_state, strict=True)
        self.criterion.load_state_dict(payload["criterion"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.epoch, self.history = epoch, deepcopy(payload["history"])
        self.best_loss, self.bad_epochs = payload["best_loss"], payload["bad_epochs"]
        self._seeded = payload["seeded"]
        self._generators = {}
        self._generator_states = payload["loader_generators"]
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"])
        if self.device.type == "cuda" and cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
