"""Software checks for untrained EEG Transformer modules; no fitting occurs."""

import io
import json

import numpy as np
import pytest
import torch
from sklearn.base import clone
from torch import nn

from metabci.brainda.algorithms.deep_learning.eeg_transformer import (
    EEGConvTransformer,
    EEGTransformer,
    make_eeg_transformer,
)


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def model():
    torch.manual_seed(27)
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
        auxiliary_dims={"emg": 5, "vision": 6},
    ).eval()


@pytest.fixture
def eeg():
    generator = torch.Generator().manual_seed(53)
    return torch.randn(3, 4, 38, generator=generator)


def _snapshot(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _assert_state_unchanged(model, previous):
    assert model.state_dict().keys() == previous.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, previous[name], rtol=0, atol=0)


def test_token_features_and_logits_have_public_shapes(model, eeg):
    """The temporal stem supports variable trial lengths without new weights."""
    before = _snapshot(model)
    with torch.no_grad():
        for samples in (11, 38, 56):
            x = eeg[..., :samples] if samples <= 38 else torch.randn(3, 4, samples)
            expected_tokens = (samples - 7 + 1 - 5) // 3 + 1
            tokens = model.tokenize(x)
            encoded = model.forward_tokens(x)
            backbone = model.cal_backbone(x)
            logits = model(x)
            assert tokens.shape == encoded.shape == (3, expected_tokens, 16)
            assert backbone.shape == (3, 16)
            assert logits.shape == (3, 3)
            for value in (tokens, encoded, backbone, logits):
                assert torch.isfinite(value).all()
            torch.testing.assert_close(backbone, encoded.mean(dim=1))
    _assert_state_unchanged(model, before)


def test_batch_examples_are_independent_and_permutation_equivariant(model, eeg):
    auxiliary = {"emg": torch.randn(3, 2, 5), "vision": torch.randn(3, 6)}
    masks = {"emg": torch.tensor([[True, False], [True, True], [False, True]])}
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        together = model(eeg, auxiliary=auxiliary, auxiliary_mask=masks)
        separately = torch.cat(
            [
                model(
                    eeg[i : i + 1],
                    auxiliary={
                        name: value[i : i + 1] for name, value in auxiliary.items()
                    },
                    auxiliary_mask={
                        name: value[i : i + 1] for name, value in masks.items()
                    },
                )
                for i in range(len(eeg))
            ]
        )
        shuffled = model(
            eeg[permutation],
            auxiliary={name: value[permutation] for name, value in auxiliary.items()},
            auxiliary_mask={name: value[permutation] for name, value in masks.items()},
        )
    torch.testing.assert_close(together, separately, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(shuffled, together[permutation], rtol=1e-5, atol=1e-6)


def test_masked_auxiliary_values_and_extra_padding_do_not_affect_eeg(model, eeg):
    valid = torch.randn(3, 2, 5)
    padding = torch.full((3, 3, 5), float("nan"))
    padded = torch.cat((valid, padding), dim=1)
    mask = torch.tensor([[True, True, False, False, False]]).expand(3, -1)
    changed_padding = padded.clone()
    changed_padding[:, 2:] = torch.tensor([float("inf"), -1e8, float("nan")])[
        None, :, None
    ]
    with torch.no_grad():
        reference = model(eeg, auxiliary={"emg": valid})
        masked = model(eeg, auxiliary={"emg": padded}, auxiliary_mask={"emg": mask})
        changed = model(
            eeg, auxiliary={"emg": changed_padding}, auxiliary_mask={"emg": mask}
        )
    assert torch.isfinite(masked).all()
    torch.testing.assert_close(masked, reference, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(changed, masked, rtol=0, atol=0)


def test_completely_missing_auxiliary_matches_fully_masked_tokens(model, eeg):
    with torch.no_grad():
        missing = model(eeg)
        explicit_empty = model(eeg, auxiliary={})
        masked = model(
            eeg,
            auxiliary={"emg": torch.full((3, 2, 5), float("nan"))},
            auxiliary_mask={"emg": torch.zeros(3, 2, dtype=torch.bool)},
        )
    torch.testing.assert_close(missing, explicit_empty, rtol=0, atol=0)
    torch.testing.assert_close(masked, missing, rtol=1e-5, atol=1e-6)


def test_noncontiguous_eeg_and_single_auxiliary_vector(model, eeg):
    noncontiguous = eeg.transpose(1, 2).contiguous().transpose(1, 2)
    assert not noncontiguous.is_contiguous()
    vectors = torch.randn(3, 5)
    with torch.no_grad():
        reference = model(eeg, auxiliary={"emg": vectors.unsqueeze(1)})
        actual = model(noncontiguous, auxiliary={"emg": vectors})
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_single_backward_reaches_stem_and_auxiliary_without_updating_weights(
    model, eeg
):
    eeg = eeg.clone().requires_grad_()
    auxiliary = {
        "emg": torch.randn(3, 2, 5, requires_grad=True),
        "vision": torch.randn(3, 1, 6, requires_grad=True),
    }
    before = _snapshot(model)
    nn.functional.cross_entropy(
        model(eeg, auxiliary=auxiliary), torch.tensor([0, 1, 2])
    ).backward()
    for value in (eeg, *auxiliary.values()):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    for features in (5, 6):
        projectors = [
            layer
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
            and layer.in_features == features
            and layer.out_features == 16
        ]
        assert len(projectors) == 1
        gradient = projectors[0].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    _assert_state_unchanged(model, before)


def test_config_and_state_dict_reconstruct_identical_inference(model, eeg):
    auxiliary = {"vision": torch.randn(3, 6)}
    config = json.loads(json.dumps(model.get_config()))
    rebuilt = EEGConvTransformer(**config).eval()
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    rebuilt.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(
            rebuilt(eeg, auxiliary=auxiliary),
            model(eeg, auxiliary=auxiliary),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "preset,dimensions", [("small", (64, 2, 4, 256)), ("large", (256, 8, 8, 1024))]
)
def test_factory_presets_have_finite_untrained_cpu_forward(preset, dimensions):
    model = make_eeg_transformer(n_channels=4, n_classes=3, preset=preset).eval()
    config = model.get_config()
    assert (
        tuple(
            config[name]
            for name in ("d_model", "n_layers", "n_heads", "dim_feedforward")
        )
        == dimensions
    )
    before = _snapshot(model)
    with torch.no_grad():
        logits = model(torch.randn(1, 4, 128))
    assert logits.shape == (1, 3) and torch.isfinite(logits).all()
    _assert_state_unchanged(model, before)


def test_factory_overrides_and_returned_config_are_independent(model):
    config = model.get_config()
    del config["n_channels"], config["n_classes"]
    rebuilt = make_eeg_transformer(4, 3, preset="large", **config).eval()
    assert rebuilt.get_config() == model.get_config()
    config["auxiliary_dims"]["emg"] = 999
    assert rebuilt.get_config()["auxiliary_dims"]["emg"] == 5


def test_skorch_initialize_and_float64_infer_do_not_fit(eeg):
    net = EEGTransformer(
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
    )
    net.initialize()
    net.module_.eval()
    assert next(net.module_.parameters()).dtype == torch.float64
    before = _snapshot(net.module_)
    with torch.no_grad():
        logits = net.infer(eeg.numpy().astype(np.float64))
    assert logits.shape == (3, 3) and logits.dtype == torch.float64
    assert torch.isfinite(logits).all()
    cloned = clone(net).initialize()
    cloned.module_.eval()
    for original, copied in zip(net.module_.parameters(), cloned.module_.parameters()):
        assert original.data_ptr() != copied.data_ptr()
    with torch.no_grad():
        torch.testing.assert_close(
            cloned.infer(eeg.numpy().astype(np.float64)), logits, rtol=0, atol=0
        )
    _assert_state_unchanged(net.module_, before)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_channels": 0},
        {"n_classes": 0},
        {"d_model": 15},
        {"n_layers": 0},
        {"n_heads": 0},
        {"dim_feedforward": 0},
        {"temporal_filters": 0},
        {"temporal_kernel": 0},
        {"pool_size": 0},
        {"pool_stride": 0},
        {"dropout": -0.1},
        {"dropout": 1.1},
        {"auxiliary_dims": {"emg": 0}},
    ],
)
def test_invalid_model_configurations_raise(overrides):
    config = {"n_channels": 4, "n_classes": 3, "d_model": 16, "n_heads": 4}
    config.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        EEGConvTransformer(**config)


def test_unknown_factory_preset_raises():
    with pytest.raises(ValueError):
        make_eeg_transformer(n_channels=4, n_classes=3, preset="unknown")


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(3, 4 * 38),
        torch.randn(3, 5, 38),
        torch.randn(3, 4, 10),
        torch.randn(0, 4, 38),
        torch.ones(3, 4, 38, dtype=torch.int64),
        torch.randn(3, 4, 38, dtype=torch.float64),
        torch.empty(3, 4, 38, device="meta"),
        torch.full((3, 4, 38), float("nan")),
    ],
)
def test_invalid_eeg_inputs_raise(model, x):
    with pytest.raises((TypeError, ValueError)):
        model(x)


@pytest.mark.parametrize(
    "auxiliary,masks",
    [
        ({"unknown": torch.randn(3, 2, 5)}, None),
        ({"emg": torch.randn(3, 2, 4)}, None),
        ({"emg": torch.randn(2, 2, 5)}, None),
        ({"emg": torch.ones(3, 2, 5, dtype=torch.int64)}, None),
        ({"emg": torch.randn(3, 2, 5, dtype=torch.float64)}, None),
        ({"emg": torch.empty(3, 2, 5, device="meta")}, None),
        ({"emg": torch.full((3, 2, 5), float("nan"))}, None),
        ({"emg": torch.randn(3, 2, 5)}, {"emg": torch.ones(3, 2)}),
        ({"emg": torch.randn(3, 2, 5)}, {"emg": torch.ones(3, 3, dtype=torch.bool)}),
        (None, {"emg": torch.ones(3, 1, dtype=torch.bool)}),
    ],
)
def test_invalid_auxiliary_inputs_raise(model, eeg, auxiliary, masks):
    with pytest.raises((TypeError, ValueError)):
        model(eeg, auxiliary=auxiliary, auxiliary_mask=masks)
