"""Training-framework component checks without fitting or optimizer steps."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from metabci.brainda.algorithms.deep_learning.eeg_transformer import EEGConvTransformer
from metabci.brainda.algorithms.deep_learning.transformer_training import (
    TrainingConfig,
    TransformerTrainer,
)


class _ScaledCrossEntropy(nn.Module):
    """A parameter-free, sample-mean custom loss with serializable settings."""

    def __init__(self, scale=2.0):
        super().__init__()
        self.scale = scale

    def get_config(self):
        return {"scale": self.scale}

    def forward(self, logits, labels):
        return self.scale * nn.functional.cross_entropy(logits, labels)


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _forbid_training_and_steps(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Software validation must not fit or take an optimizer step")

    monkeypatch.setattr(TransformerTrainer, "fit", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)


def _model():
    return EEGConvTransformer(
        n_channels=4,
        n_classes=3,
        d_model=16,
        n_layers=1,
        n_heads=4,
        dim_feedforward=32,
        temporal_filters=8,
        temporal_kernel=7,
        pool_size=5,
        pool_stride=3,
        dropout=0.0,
        auxiliary_dims={"emg": 5},
    ).eval()


@pytest.fixture
def trainer(tmp_path):
    torch.manual_seed(39)
    return TransformerTrainer(
        _model(),
        TrainingConfig(
            epochs=4,
            batch_size=3,
            warmup_epochs=1,
            patience=2,
            output_dir=str(tmp_path / "not-created"),
        ),
    )


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(73)
    auxiliary = torch.randn(3, 2, 5, generator=generator, dtype=torch.float64)
    auxiliary[:, 1] = float("nan")
    return {
        "X": torch.randn(3, 4, 38, generator=generator, dtype=torch.float64),
        "y": torch.tensor([0, 1, 2], dtype=torch.long),
        "auxiliary": {"emg": auxiliary},
        "auxiliary_mask": {"emg": torch.tensor([[True, False]]).expand(3, -1)},
    }


def _snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_state_unchanged(model, before):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def test_config_is_json_serializable_and_roundtrips():
    config = TrainingConfig(epochs=8, warmup_epochs=2, lr=2e-4, output_dir="run-folder")
    encoded = json.loads(json.dumps(config.to_dict()))
    restored = TrainingConfig.from_dict(encoded)
    assert restored == config


def test_constructor_creates_components_without_training_or_output_files(trainer):
    assert trainer.epoch == 0 and trainer.history == []
    assert trainer.optimizer.state == {}
    assert not trainer.model.training
    assert not Path(trainer.config.output_dir).exists()


def test_forward_and_loss_cast_features_but_preserve_batch_and_parameters(
    trainer, batch
):
    before = _snapshot(trainer.model)
    with torch.no_grad():
        actual = trainer.forward_batch(batch)
        expected = trainer.model(
            batch["X"].float(),
            auxiliary={"emg": batch["auxiliary"]["emg"].float()},
            auxiliary_mask=batch["auxiliary_mask"],
        )
        loss = trainer.compute_loss(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(loss, nn.functional.cross_entropy(expected, batch["y"]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert batch["X"].dtype == batch["auxiliary"]["emg"].dtype == torch.float64
    assert batch["auxiliary_mask"]["emg"].dtype == torch.bool
    assert torch.isnan(batch["auxiliary"]["emg"][:, 1]).all()
    _assert_state_unchanged(trainer.model, before)
    assert trainer.optimizer.state == {}


def test_loss_backward_has_gradients_without_parameter_or_buffer_updates(
    trainer, batch
):
    batch["X"].requires_grad_()
    batch["auxiliary"]["emg"].requires_grad_()
    before = _snapshot(trainer.model)
    loss = trainer.compute_loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for value in (batch["X"], batch["auxiliary"]["emg"]):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    torch.testing.assert_close(
        batch["auxiliary"]["emg"].grad[:, 1], torch.zeros(3, 5, dtype=torch.float64)
    )
    gradients = [parameter.grad for parameter in trainer.model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    _assert_state_unchanged(trainer.model, before)
    assert trainer.optimizer.state == {}


def test_evaluate_reports_sample_weighted_metrics_without_learning(trainer, batch):
    batches = [
        {
            "X": batch["X"][:2],
            "y": batch["y"][:2],
            "auxiliary": {"emg": batch["auxiliary"]["emg"][:2]},
            "auxiliary_mask": {"emg": batch["auxiliary_mask"]["emg"][:2]},
        },
        {
            "X": batch["X"][2:],
            "y": batch["y"][2:],
            "auxiliary": {"emg": batch["auxiliary"]["emg"][2:]},
            "auxiliary_mask": {"emg": batch["auxiliary_mask"]["emg"][2:]},
        },
    ]
    before = _snapshot(trainer.model)
    with torch.no_grad():
        logits = trainer.forward_batch(batch)
        expected_loss = nn.functional.cross_entropy(logits, batch["y"]).item()
        expected_accuracy = (logits.argmax(dim=1) == batch["y"]).double().mean().item()
    actual = trainer.evaluate(batches)
    assert actual["samples"] == 3
    assert actual["loss"] == pytest.approx(expected_loss, rel=1e-6)
    assert actual["accuracy"] == pytest.approx(expected_accuracy)
    assert not trainer.model.training
    _assert_state_unchanged(trainer.model, before)
    assert trainer.optimizer.state == {}


def test_checkpoint_restores_untrained_model_components_and_rng(
    trainer, batch, tmp_path
):
    path = tmp_path / "untrained.pt"
    before = _snapshot(trainer.model)
    random.seed(14)
    np.random.seed(14)
    torch.manual_seed(14)
    saved = trainer.save_checkpoint(path)
    assert saved == path and path.exists()
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    restored = TransformerTrainer(_model(), trainer.config)
    restored.model.eval()
    restored.optimizer.param_groups[0]["lr"] = 0.4
    restored.load_checkpoint(path, trusted=True)
    assert (random.random(), np.random.random()) == expected_rng[:2]
    torch.testing.assert_close(torch.rand(3), expected_rng[2], rtol=0, atol=0)
    assert restored.optimizer.state_dict() == trainer.optimizer.state_dict()
    assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
    assert restored.epoch == trainer.epoch == 0
    assert restored.history == trainer.history == []
    assert restored.best_loss == trainer.best_loss
    assert restored.bad_epochs == trainer.bad_epochs
    with torch.no_grad():
        torch.testing.assert_close(
            restored.forward_batch(batch), trainer.forward_batch(batch), rtol=0, atol=0
        )
    _assert_state_unchanged(trainer.model, before)
    _assert_state_unchanged(restored.model, before)


def test_checkpoint_requires_explicit_trust_and_rejects_mismatched_config(
    trainer, tmp_path
):
    path = trainer.save_checkpoint(tmp_path / "untrained.pt")
    before = _snapshot(trainer.model)
    with pytest.raises(ValueError):
        trainer.load_checkpoint(path)
    _assert_state_unchanged(trainer.model, before)
    incompatible = TransformerTrainer(
        _model(),
        TrainingConfig(epochs=9, warmup_epochs=1, patience=2),
    )
    incompatible.model.eval()
    other_before = _snapshot(incompatible.model)
    with pytest.raises(ValueError):
        incompatible.load_checkpoint(path, trusted=True)
    _assert_state_unchanged(incompatible.model, other_before)


def test_checkpoint_roundtrips_synthetic_nonempty_resume_components(trainer, tmp_path):
    """Populate serializer fixtures directly; these are not trained moments."""
    before = _snapshot(trainer.model)
    parameter = next(trainer.model.parameters())
    trainer.optimizer.state[parameter] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.full_like(parameter, 0.01),
        "exp_avg_sq": torch.full_like(parameter, 0.02),
    }
    schedule = trainer.scheduler.state_dict()
    schedule["last_epoch"] = 1
    trainer.scheduler.load_state_dict(schedule)
    trainer.epoch = 1
    trainer.history = [{"fixture": "synthetic metadata; no parameter update"}]
    trainer.best_loss = 0.8
    trainer.bad_epochs = 1
    path = trainer.save_checkpoint(tmp_path / "synthetic-component-state.pt")
    restored = TransformerTrainer(_model(), trainer.config)
    restored.model.eval()
    restored.load_checkpoint(path, trusted=True)
    _assert_nested_equal(
        restored.optimizer.state_dict(), trainer.optimizer.state_dict()
    )
    _assert_nested_equal(
        restored.scheduler.state_dict(), trainer.scheduler.state_dict()
    )
    assert restored.epoch == 1 and restored.history == trainer.history
    assert restored.best_loss == 0.8 and restored.bad_epochs == 1
    _assert_state_unchanged(trainer.model, before)
    _assert_state_unchanged(restored.model, before)


def test_checkpoint_rejects_different_attention_config_before_mutating_weights(
    trainer, tmp_path
):
    path = trainer.save_checkpoint(tmp_path / "untrained.pt")
    config = trainer.model.get_config()
    config["n_heads"] = 2
    other = TransformerTrainer(EEGConvTransformer(**config).eval(), trainer.config)
    before = _snapshot(other.model)
    with pytest.raises(ValueError):
        other.load_checkpoint(path, trusted=True)
    _assert_state_unchanged(other.model, before)


def test_weighted_cross_entropy_evaluation_uses_global_target_weight_sum(
    trainer, batch
):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 9.0]))
    weighted = TransformerTrainer(trainer.model, trainer.config, criterion=criterion)
    weighted.model.eval()
    before = _snapshot(weighted.model)
    split = [
        {
            "X": batch["X"][indices],
            "y": batch["y"][indices],
            "auxiliary": {"emg": batch["auxiliary"]["emg"][indices]},
            "auxiliary_mask": {"emg": batch["auxiliary_mask"]["emg"][indices]},
        }
        for indices in (slice(0, 2), slice(2, 3))
    ]
    with torch.no_grad():
        logits = weighted.forward_batch(batch)
        expected = criterion(logits, batch["y"])
        actual_loss = weighted.compute_loss(batch)
    torch.testing.assert_close(actual_loss, expected, rtol=0, atol=0)
    assert weighted.evaluate(split)["loss"] == pytest.approx(expected.item(), rel=1e-6)
    _assert_state_unchanged(weighted.model, before)


def test_checkpoint_rejects_different_loss_weights(trainer, tmp_path):
    weighted = TransformerTrainer(
        trainer.model,
        trainer.config,
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0])),
    )
    path = weighted.save_checkpoint(tmp_path / "weighted-untrained.pt")
    other = TransformerTrainer(
        _model(),
        trainer.config,
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 4.0])),
    )
    other.model.eval()
    before = _snapshot(other.model)
    with pytest.raises(ValueError):
        other.load_checkpoint(path, trusted=True)
    _assert_state_unchanged(other.model, before)


def test_custom_scalar_loss_and_checkpoint_roundtrip(trainer, batch, tmp_path):
    custom = TransformerTrainer(
        trainer.model, trainer.config, criterion=_ScaledCrossEntropy(scale=2.5)
    )
    before = _snapshot(custom.model)
    with torch.no_grad():
        logits = custom.forward_batch(batch)
        expected = 2.5 * nn.functional.cross_entropy(logits, batch["y"])
        torch.testing.assert_close(custom.compute_loss(batch), expected, rtol=0, atol=0)
    assert custom.evaluate([batch])["loss"] == pytest.approx(expected.item())
    path = custom.save_checkpoint(tmp_path / "custom-loss-untrained.pt")
    restored = TransformerTrainer(
        _model(), custom.config, criterion=_ScaledCrossEntropy(scale=2.5)
    )
    restored.model.eval()
    restored.load_checkpoint(path, trusted=True)
    with torch.no_grad():
        torch.testing.assert_close(
            restored.compute_loss(batch), expected, rtol=0, atol=0
        )
    incompatible = TransformerTrainer(
        _model(), custom.config, criterion=_ScaledCrossEntropy(scale=1.0)
    )
    incompatible.model.eval()
    with pytest.raises(ValueError):
        incompatible.load_checkpoint(path, trusted=True)
    _assert_state_unchanged(custom.model, before)
    _assert_state_unchanged(restored.model, before)


def test_zero_target_weight_batch_is_rejected(trainer, batch):
    weighted = TransformerTrainer(
        trainer.model,
        trainer.config,
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.0, 0.0, 1.0])),
    )
    batch["y"] = torch.tensor([0, 1, 1], dtype=torch.long)
    with pytest.raises(ValueError):
        weighted.compute_loss(batch)


@pytest.mark.parametrize("reduction", ["sum", "none"])
def test_non_mean_cross_entropy_is_rejected(trainer, reduction):
    with pytest.raises(ValueError):
        TransformerTrainer(
            trainer.model,
            trainer.config,
            criterion=nn.CrossEntropyLoss(reduction=reduction),
        )


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([-1.0, 1.0, 1.0]),
        torch.tensor([float("nan"), 1.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        torch.zeros(3),
    ],
)
def test_invalid_cross_entropy_weights_are_rejected(trainer, batch, weights):
    with pytest.raises(ValueError):
        weighted = TransformerTrainer(
            trainer.model,
            trainer.config,
            criterion=nn.CrossEntropyLoss(weight=weights),
        )
        weighted.model.eval()
        weighted.compute_loss(batch)


@pytest.mark.parametrize("labels", [torch.tensor([-1, 1, 2]), torch.tensor([0, 1, 3])])
def test_out_of_range_class_labels_are_rejected(trainer, batch, labels):
    batch["y"] = labels
    with pytest.raises(ValueError):
        trainer.compute_loss(batch)


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"lr": 0},
        {"lr": float("nan")},
        {"weight_decay": -1},
        {"warmup_epochs": -1},
        {"clip_grad_norm": 0},
        {"patience": 0},
        {"min_delta": -1},
    ],
)
def test_invalid_training_configurations_raise(overrides):
    with pytest.raises((TypeError, ValueError)):
        TrainingConfig(**overrides)


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0, 1, 2], dtype=torch.int32)],
)
def test_non_long_labels_are_rejected(trainer, batch, labels):
    batch["y"] = labels
    with pytest.raises((TypeError, ValueError)):
        trainer.compute_loss(batch)


def test_non_boolean_mask_is_rejected(trainer, batch):
    batch["auxiliary_mask"]["emg"] = torch.ones(3, 2)
    with pytest.raises((TypeError, ValueError)):
        trainer.forward_batch(batch)
