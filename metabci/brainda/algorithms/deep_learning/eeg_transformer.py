"""Untrained convolutional EEG Transformers with optional auxiliary features.

This independent implementation uses a Conformer-inspired convolutional stem
and native PyTorch attention. It provides no pretrained weights or claims about
classification accuracy. Auxiliary inputs are user-supplied feature vectors;
this module does not encode images/text or establish cross-modal alignment.
"""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
from torch import nn

from .base import SkorchNet


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("{} must be a positive integer.".format(name))
    return int(value)


class EEGConvTransformer(nn.Module):
    """A convolutional EEG encoder and classifier without pretrained weights.

    Parameters
    ----------
    n_channels, n_classes : int
        Fixed EEG channel count/order and number of output classes.
    d_model, n_layers, n_heads, dim_feedforward : int
        Transformer width, depth, attention heads, and feed-forward width.
        ``d_model`` must be divisible by ``n_heads``.
    temporal_filters, temporal_kernel, pool_size, pool_stride : int
        Stem dimensions. Both convolutions and pooling use valid padding.
    dropout : float
        Dropout probability in [0, 1).
    auxiliary_dims : mapping of str to int, optional
        Feature width for each permitted auxiliary modality. Names are ordered
        lexicographically so configuration/checkpoint reconstruction is stable.

    Notes
    -----
    EEG input is a finite floating tensor ``X`` of shape (B, C, T). A batch must
    share its unpadded T; different calls may have different T. EEG padding and
    lengths are not supported. T must be at least
    ``temporal_kernel + pool_size - 1`` (99 with defaults). Channel order,
    sampling rate, units, and any normalization are the caller's responsibility.

    Auxiliary features are supplied as ``{name: tensor}``, with shape (B, M, F)
    or (B, F) for one token. Optional boolean masks have shape (B, M), where True
    means valid. Configured modalities may be omitted; unknown names are errors.
    Masked feature values, including NaN/Inf, are cleared before projection.
    Valid features must be finite. Features must share X's device and dtype;
    masks must share its device. Only EEG tokens contribute to the readout.

    Sinusoidal positions encode EEG token order. Auxiliary tokens have learned
    modality embeddings, but no assumed time/spatial ordering. If such ordering
    is needed, the user-supplied features must encode it. Attention permits
    within-trial use of all supplied tokens and is not a causal online decoder.
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=256,
        temporal_filters=32,
        temporal_kernel=25,
        pool_size=75,
        pool_stride=15,
        dropout=0.1,
        auxiliary_dims=None,
    ):
        super().__init__()
        dimensions = {
            "n_channels": n_channels,
            "n_classes": n_classes,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "dim_feedforward": dim_feedforward,
            "temporal_filters": temporal_filters,
            "temporal_kernel": temporal_kernel,
            "pool_size": pool_size,
            "pool_stride": pool_stride,
        }
        for name, value in dimensions.items():
            setattr(self, name, _positive_int(value, name))
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be a finite probability in [0, 1).")
        self.dropout = float(dropout)
        if auxiliary_dims is None:
            auxiliary_dims = {}
        if not isinstance(auxiliary_dims, Mapping):
            raise TypeError("auxiliary_dims must be a mapping of names to widths.")
        if any(not isinstance(name, str) or not name for name in auxiliary_dims):
            raise ValueError("Auxiliary modality names must be nonempty strings.")
        self.auxiliary_dims = {
            name: _positive_int(auxiliary_dims[name], "auxiliary_dims[{}]".format(name))
            for name in sorted(auxiliary_dims)
        }
        self.minimum_samples = self.temporal_kernel + self.pool_size - 1

        self.stem = nn.Sequential(
            nn.Conv2d(1, self.temporal_filters, (1, self.temporal_kernel)),
            nn.Conv2d(
                self.temporal_filters,
                self.temporal_filters,
                (self.n_channels, 1),
            ),
            nn.BatchNorm2d(self.temporal_filters),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size), stride=(1, self.pool_stride)),
            nn.Dropout(self.dropout),
        )
        self.projection = nn.Linear(self.temporal_filters, self.d_model)
        self.auxiliary_projections = nn.ModuleList(
            nn.Linear(width, self.d_model) for width in self.auxiliary_dims.values()
        )
        self.modality_embeddings = nn.ParameterList(
            nn.Parameter(torch.empty(self.d_model)) for _ in self.auxiliary_dims
        )
        for embedding in self.modality_embeddings:
            nn.init.normal_(embedding, std=0.02)
        # Sequence-first layers avoid batch_first/norm_first APIs added after
        # PyTorch 1.7. Construct each layer separately for independent weights.
        self.transformer_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                self.d_model,
                self.n_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation="gelu",
            )
            for _ in range(self.n_layers)
        )
        self.final_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.n_classes)

    def get_config(self):
        """Return a fresh JSON-compatible constructor configuration, not weights."""
        names = (
            "n_channels",
            "n_classes",
            "d_model",
            "n_layers",
            "n_heads",
            "dim_feedforward",
            "temporal_filters",
            "temporal_kernel",
            "pool_size",
            "pool_stride",
            "dropout",
        )
        config = {name: getattr(self, name) for name in names}
        config["auxiliary_dims"] = dict(self.auxiliary_dims)
        return config

    def _validate_eeg(self, X):
        if not isinstance(X, torch.Tensor) or not torch.is_floating_point(X):
            raise TypeError("X must be a floating-point torch.Tensor.")
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, channels, samples).")
        if X.shape[0] == 0 or X.shape[1] != self.n_channels:
            raise ValueError("X needs a nonempty batch and the configured n_channels.")
        if X.shape[2] < self.minimum_samples:
            raise ValueError(
                "X needs at least {} samples; EEG padding/lengths are unsupported.".format(
                    self.minimum_samples
                )
            )
        weight = self.projection.weight
        if X.device != weight.device or X.dtype != weight.dtype:
            raise ValueError("X must match the model's device and floating dtype.")
        if not torch.isfinite(X).all().item():
            raise ValueError("EEG input X must contain only finite values.")

    def _position_encoding(self, length, reference):
        positions = torch.arange(
            length, dtype=reference.dtype, device=reference.device
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0, self.d_model, 2, dtype=reference.dtype, device=reference.device
            )
            * (-math.log(10000.0) / self.d_model)
        )
        encoding = reference.new_zeros((length, self.d_model))
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: self.d_model // 2])
        return encoding

    def tokenize(self, X):
        """Return EEG stem/projection/position tokens of shape (B, L, D)."""
        self._validate_eeg(X)
        features = self.stem(X.unsqueeze(1)).squeeze(2).transpose(1, 2)
        tokens = self.projection(features)
        return tokens + self._position_encoding(tokens.shape[1], tokens).unsqueeze(0)

    def _joint_tokens(self, eeg_tokens, auxiliary, auxiliary_mask):
        if auxiliary is None:
            auxiliary = {}
        if auxiliary_mask is None:
            auxiliary_mask = {}
        if not isinstance(auxiliary, Mapping) or not isinstance(
            auxiliary_mask, Mapping
        ):
            raise TypeError("auxiliary and auxiliary_mask must be mappings or None.")
        unknown = set(auxiliary) - set(self.auxiliary_dims)
        if unknown:
            raise ValueError(
                "Unknown auxiliary modality: {!r}.".format(next(iter(unknown)))
            )
        if set(auxiliary_mask) - set(auxiliary):
            raise ValueError("Every auxiliary mask must have corresponding features.")
        batch_size = eeg_tokens.shape[0]
        token_groups = [eeg_tokens]
        valid_groups = [
            torch.ones(eeg_tokens.shape[:2], dtype=torch.bool, device=eeg_tokens.device)
        ]
        for index, (name, width) in enumerate(self.auxiliary_dims.items()):
            if name not in auxiliary:
                continue
            features = auxiliary[name]
            if not isinstance(features, torch.Tensor) or not torch.is_floating_point(
                features
            ):
                raise TypeError("Auxiliary features must be floating-point tensors.")
            if features.ndim == 2:
                features = features.unsqueeze(1)
            if (
                features.ndim != 3
                or features.shape[0] != batch_size
                or features.shape[1] == 0
                or features.shape[2] != width
            ):
                raise ValueError(
                    "Auxiliary {!r} must have shape (B, M, {}) with M > 0.".format(
                        name, width
                    )
                )
            if (
                features.dtype != eeg_tokens.dtype
                or features.device != eeg_tokens.device
            ):
                raise ValueError("Auxiliary features must match X's device and dtype.")
            valid = auxiliary_mask.get(name)
            if name in auxiliary_mask:
                if (
                    not isinstance(valid, torch.Tensor)
                    or valid.dtype != torch.bool
                    or valid.shape != features.shape[:2]
                    or valid.device != features.device
                ):
                    raise ValueError(
                        "Auxiliary masks must be boolean (B, M) tensors on X's device."
                    )
            else:
                valid = torch.ones(
                    features.shape[:2], dtype=torch.bool, device=features.device
                )
            clean_features = torch.where(
                valid.unsqueeze(-1), features, torch.zeros_like(features)
            )
            if not torch.isfinite(clean_features).all().item():
                raise ValueError(
                    "Valid auxiliary features must contain only finite values."
                )
            projected = self.auxiliary_projections[index](clean_features)
            projected = projected + self.modality_embeddings[index].view(1, 1, -1)
            token_groups.append(projected)
            valid_groups.append(valid)
        return torch.cat(token_groups, dim=1), ~torch.cat(valid_groups, dim=1)

    def forward_tokens(self, X, auxiliary=None, auxiliary_mask=None):
        """Return final-normalized EEG tokens (B, L, D) after joint attention.

        Auxiliary tokens participate only where their validity masks are True.
        The returned sequence contains only the original EEG token positions.
        """
        eeg_tokens = self.tokenize(X)
        tokens, padding_mask = self._joint_tokens(eeg_tokens, auxiliary, auxiliary_mask)
        sequence = tokens.transpose(0, 1)
        for layer in self.transformer_layers:
            sequence = layer(sequence, src_key_padding_mask=padding_mask)
        eeg_sequence = sequence[: eeg_tokens.shape[1]].transpose(0, 1)
        return self.final_norm(eeg_sequence)

    def cal_backbone(self, X, auxiliary=None, auxiliary_mask=None):
        """Return (B, D) pooled EEG features for MetaBCI fine-tuning heads."""
        return self.forward_tokens(X, auxiliary, auxiliary_mask).mean(dim=1)

    def forward(self, X, auxiliary=None, auxiliary_mask=None):
        """Return raw classification logits (B, n_classes), without softmax."""
        return self.classifier(self.cal_backbone(X, auxiliary, auxiliary_mask))


def make_eeg_transformer(
    n_channels, n_classes, preset="small", auxiliary_dims=None, **overrides
):
    """Create an untrained raw module from a preset and constructor overrides.

    ``small`` uses width/depth/heads/feed-forward = 64/2/4/256;
    ``large`` uses 256/8/8/1024. Neither preset supplies trained weights.
    """
    presets = {
        "small": dict(d_model=64, n_layers=2, n_heads=4, dim_feedforward=256),
        "large": dict(d_model=256, n_layers=8, n_heads=8, dim_feedforward=1024),
    }
    if not isinstance(preset, str) or preset not in presets:
        raise ValueError("preset must be 'small' or 'large'.")
    config = dict(presets[preset])
    config.update(overrides)
    return EEGConvTransformer(
        n_channels=n_channels,
        n_classes=n_classes,
        auxiliary_dims=auxiliary_dims,
        **config,
    )


@SkorchNet
class EEGTransformer(EEGConvTransformer):
    """MetaBCI skorch classifier wrapper using the existing training defaults.

    Construction returns a ``NeuralNetClassifierNoLog`` (double precision, CPU).
    Configure training options on that returned estimator using ``set_params``.
    This wrapper does not start training until the caller invokes ``fit``.
    """
