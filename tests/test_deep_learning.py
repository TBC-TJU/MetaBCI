# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for deep learning models
#
# Tests for: metabci.brainda.algorithms.deep_learning

import numpy as np
import torch

from metabci.brainda.algorithms.deep_learning.deepnet import Deep4Net
from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet
from metabci.brainda.algorithms.deep_learning.shallownet import ShallowNet

# ============================================================================
# Constants
# ============================================================================

N_CHANNELS = 8
N_SAMPLES = 250
N_CLASSES = 2
BATCH_SIZE = 4
# Deep4Net needs larger temporal dimension due to 4 conv-pool stages
# with kernel_size=10 and pool_stride=3. Minimum ~387 samples.
DEEP4_N_SAMPLES = 1000


# ============================================================================
# Tests for EEGNet
# ============================================================================


class TestEEGNet:
    """Tests for the EEGNet architecture.

    Note: EEGNet.forward() calls X.unsqueeze(1), so input should be
    3D: (batch, n_channels, n_samples), NOT (batch, 1, n_channels, n_samples).
    The SkorchNet wrapper (accessed via EEGNet()) returns a NeuralNetClassifier.
    To test the raw nn.Module forward pass, we access model.module if wrapped.
    """

    def test_instantiation(self):
        """Test that EEGNet can be instantiated."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert model is not None

    def test_forward_pass_shape(self):
        """Test forward pass output shape.

        EEGNet is wrapped by SkorchNet, so EEGNet(...) returns a skorch
        NeuralNetClassifier. The actual nn.Module is model.module_.
        However, model.module_ is only available after fit. For direct
        forward pass testing, we import the raw class and test it.
        """
        # Access the raw nn.Module class (before SkorchNet wrapping)
        # The module is stored in the eegnet.py file as a class inheriting nn.Module
        # When imported directly from the file, it IS the SkorchNet wrapper due to __init__.py
        # So we need to get the underlying module from the wrapper

        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)

        # model is a NeuralNetClassifier (skorch). Get underlying module:
        if hasattr(model, "module"):
            # Before fit, module is the class, not instance
            # We need to instantiate it or use initialize()
            model.initialize()
            net = model.module_
        else:
            net = model

        net = net.float()
        net.eval()
        # Input should be 3D: (batch, channels, samples)
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES).float()
        with torch.no_grad():
            out = net(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_output_finite(self):
        """Test that output contains no NaN/Inf."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        if hasattr(model, "module"):
            model.initialize()
            net = model.module_
        else:
            net = model
        net = net.float()
        net.eval()
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES).float()
        with torch.no_grad():
            out = net(x)
        assert torch.all(torch.isfinite(out))

    def test_different_n_classes(self):
        """Test EEGNet with different number of classes."""
        for n_classes in [2, 4, 10]:
            model = EEGNet(
                n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=n_classes
            )
            if hasattr(model, "module"):
                model.initialize()
                net = model.module_
            else:
                net = model
            net = net.float()
            net.eval()
            x = torch.randn(2, N_CHANNELS, N_SAMPLES).float()
            with torch.no_grad():
                out = net(x)
            assert out.shape == (2, n_classes)


# ============================================================================
# Tests for ShallowNet
# ============================================================================


class TestShallowNet:
    """Tests for the ShallowNet architecture.

    Note: ShallowNet.forward() calls X.unsqueeze(1), so input should be
    3D: (batch, n_channels, n_samples).
    """

    def test_instantiation(self):
        model = ShallowNet(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        assert model is not None

    def test_forward_pass_shape(self):
        model = ShallowNet(
            n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES
        )
        if hasattr(model, "module"):
            model.initialize()
            net = model.module_
        else:
            net = model
        net = net.float()
        net.eval()
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_SAMPLES).float()
        with torch.no_grad():
            out = net(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)


# ============================================================================
# Tests for Deep4Net
# ============================================================================


class TestDeep4Net:
    """Tests for the Deep4Net architecture.

    Note: Deep4Net has Ensure4d which adds trailing dimensions.
    It expects input shape (batch, n_channels, n_samples, 1).
    It requires larger n_samples (>=387) due to 4 conv-pool stages.
    """

    def test_instantiation(self):
        model = Deep4Net(
            n_channels=N_CHANNELS, n_samples=DEEP4_N_SAMPLES, n_classes=N_CLASSES
        )
        assert model is not None

    def test_forward_pass_shape(self):
        model = Deep4Net(
            n_channels=N_CHANNELS, n_samples=DEEP4_N_SAMPLES, n_classes=N_CLASSES
        )
        if hasattr(model, "module"):
            model.initialize()
            net = model.module_
        else:
            net = model
        net = net.float()
        net.eval()
        # Deep4Net uses Ensure4d which adds trailing dims if needed
        x = torch.randn(BATCH_SIZE, N_CHANNELS, DEEP4_N_SAMPLES, 1).float()
        with torch.no_grad():
            out = net(x)
        assert out.shape[0] == BATCH_SIZE
        assert out.shape[-1] == N_CLASSES

    def test_output_finite(self):
        model = Deep4Net(
            n_channels=N_CHANNELS, n_samples=DEEP4_N_SAMPLES, n_classes=N_CLASSES
        )
        if hasattr(model, "module"):
            model.initialize()
            net = model.module_
        else:
            net = model
        net = net.float()
        net.eval()
        x = torch.randn(BATCH_SIZE, N_CHANNELS, DEEP4_N_SAMPLES, 1).float()
        with torch.no_grad():
            out = net(x)
        assert torch.all(torch.isfinite(out))


# ============================================================================
# Tests for sklearn-compatible interface (skorch wrapper)
# ============================================================================


class TestSkorchInterface:
    """Tests for the skorch NeuralNetClassifier wrapper.

    MetaBCI wraps models via SkorchNet(Module) which returns a
    NeuralNetClassifierNoLog skorch wrapper with fit/predict methods.
    The model params (n_channels, etc.) go to SkorchNet.__call__,
    while training params (max_epochs, etc.) must be set on the wrapper
    via .set_params() AFTER instantiation.
    """

    def test_eegnet_has_fit_method(self):
        """EEGNet wrapped by skorch should have fit/predict methods."""
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_eegnet_fit_predict(self):
        """Test that EEGNet can fit on small synthetic data and predict.

        SkorchNet wraps EEGNet in NeuralNetClassifierNoLog. We configure
        training params (max_epochs, batch_size) on the wrapper object after
        instantiation, not via EEGNet constructor.
        """
        model = EEGNet(n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=N_CLASSES)
        # Configure skorch training parameters via set_params (not constructor)
        model.set_params(max_epochs=2, batch_size=8, verbose=0)

        # SkorchNet calls model.double(), so data should be float64
        rng = np.random.RandomState(42)
        X_train = rng.randn(20, N_CHANNELS, N_SAMPLES).astype(np.float64)
        y_train = np.array([0] * 10 + [1] * 10).astype(np.int64)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        assert y_pred.shape == y_train.shape
        assert set(y_pred).issubset({0, 1})
