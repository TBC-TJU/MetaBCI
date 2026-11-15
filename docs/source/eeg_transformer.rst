Untrained EEG Transformer and feature fusion
============================================

This extension provides independently implemented, Conformer-inspired EEG
classification modules and an explicit single-device training recipe. It ships
architecture code and initialization, without pretrained weights or an EEG
accuracy claim. It does not reproduce the published EEG-Conformer architecture
or its benchmark results.

Model presets
-------------

Both presets use the same implementation. The small preset is the default;
the large preset is available for users with suitable training hardware.

.. list-table:: Default configurations, 22 EEG channels and 4 classes
   :header-rows: 1

   * - Preset
     - Width / layers / heads / feed-forward width
     - Parameters, EEG only
   * - small
     - 64 / 2 / 4 / 256
     - 125,924
   * - large
     - 256 / 8 / 8 / 1024
     - 6,351,524

The stem uses temporal convolution, full-channel spatial convolution,
batch normalization, ELU and average pooling. Projected tokens receive
sinusoidal positions and pass through Transformer encoder layers. The mean of
the final EEG tokens becomes a feature vector; a linear head returns class
logits. A default 1,000-sample window produces 61 EEG tokens. These are tensor
tokens for attention, rather than discrete vocabulary IDs.

Input and output contracts
--------------------------

* EEG ``X`` is a floating tensor ``(B, C, T)`` with a fixed channel montage/order.
  Every window in a batch has the same unpadded length. Different calls can use
  different lengths; the default minimum is 99 samples. EEG padding masks and
  online causal decoding are not implemented.
* ``auxiliary`` is an optional dictionary of named, already encoded numeric
  features. Each tensor has shape ``(B, M, F)`` or ``(B, F)`` for one token.
  Feature widths are specified in ``auxiliary_dims`` at model construction.
  Multiple named modalities are supported; configured modalities can be omitted.
* ``auxiliary_mask`` contains boolean ``(B, M)`` masks, with **True meaning
  valid**. Invalid feature values are zeroed before projection, so masked NaN
  placeholders cannot contaminate the forward pass. EEG and valid features
  must be finite. Inputs must match the raw model's device and floating dtype.
* ``tokenize(X)`` returns ``(B, L, D)`` EEG stem/position tokens.
  ``forward_tokens(...)`` returns the final ``(B, L, D)`` EEG tokens after
  joint attention; auxiliary tokens are excluded from the returned sequence.
  ``cal_backbone(...)`` returns pooled ``(B, D)`` features, and ``forward(...)``
  returns raw ``(B, K)`` logits. Targets are separate integer class indices.

The auxiliary feature encoder, units, normalization, sampling rates and window
boundaries are supplied by the user. Auxiliary tokens have modality embeddings
but no built-in temporal/spatial positional order; encode any required ordering
in their input features. Raw text, images, audio and arbitrary sensors need
their own encoders. Providing features permits joint attention; it does not
establish learned semantic alignment before training.

Sampling rate changes the physical duration covered by the convolution and
pooling kernels. Record it with the channel order, units and preprocessing in
the experiment configuration. Fit data-dependent normalization or external
encoders using training data only. Auxiliary inputs must also be available at
prediction time; neither targets nor post-outcome information are features.

Raw module and MetaBCI estimator
-------------------------------

.. code-block:: python

   import torch
   from metabci.brainda.algorithms.deep_learning import (
       EEGTransformer, make_eeg_transformer,
   )

   model = make_eeg_transformer(
       n_channels=22, n_classes=4, preset="small",
       auxiliary_dims={"emg": 16, "context": 8},
   )
   model.eval()
   X = torch.randn(2, 22, 1000)
   with torch.no_grad():
       logits = model(X, auxiliary={"context": torch.randn(2, 8)})
       features = model.cal_backbone(X)  # EEG-only is also valid

   # Existing skorch estimator interface; initialization alone does not fit.
   estimator = EEGTransformer(n_channels=22, n_classes=4)
   estimator.initialize()
   # The existing wrapper uses float64 and its own training defaults.
   # Users can configure it with estimator.set_params(...) and later call fit.

Use ``make_eeg_transformer(..., preset="large")`` for the larger configuration.
Constructor overrides such as ``n_layers`` and ``temporal_kernel`` are accepted.
``get_config()`` returns the complete JSON-compatible raw-model configuration.
The explicit trainer below uses the raw module; it is a separate interface from
the existing skorch estimator.

Trial pairing and NPZ data interface
-----------------------------------

``EEGFeatureDataset`` pairs complete feature records by unique string trial IDs,
not by row order or class label. Each supplied modality needs its own IDs.
Missing records become all-invalid tokens, while duplicate IDs or records
referring to unknown EEG trials are rejected. IDs must describe the same
trial/window. This join does not resample signals or synchronize sensor clocks.

``load_feature_dataset`` reads NPZ archives with these fields:

.. list-table:: Array fields
   :header-rows: 1

   * - Field
     - Shape and meaning
   * - ``X``
     - ``(N, C, T)`` EEG windows
   * - ``y``
     - ``(N,)`` integer labels in ``[0, K)``
   * - ``pair_ids``
     - ``(N,)`` unique Unicode strings; required with auxiliary features
   * - ``aux__NAME``
     - ``(R, M, F)`` or ``(R, F)`` numeric features in their own record order
   * - ``ids__NAME``
     - ``(R,)`` corresponding unique Unicode trial IDs; required for that modality
   * - ``mask__NAME``
     - Optional boolean ``(R, M)`` validity mask

Do not use object arrays; loading disables pickle. Field names are checked and
modality names cannot contain ``__``. Datasets store CPU float32 tensors by
default; the trainer transfers each batch to its configured device/dtype.

.. code-block:: python

   import numpy as np

   # X and y have already been prepared by the experiment's data pipeline.
   # EEG-only forward inspection needs just X and y.
   np.savez("train.npz", X=X.astype(np.float32), y=y.astype(np.int64))

   # Features may be reordered or missing for some trials.
   np.savez(
       "paired_train.npz", X=X.astype(np.float32), y=y.astype(np.int64),
       pair_ids=trial_ids, aux__emg=emg_features,
       ids__emg=emg_trial_ids, mask__emg=emg_valid_tokens,
   )

MetaBCI paradigms already return compatible EEG arrays and metadata:

.. code-block:: python

   import json
   import numpy as np
   from metabci.brainda.datasets import BNCI2014001
   from metabci.brainda.paradigms import MotorImagery

   dataset = BNCI2014001()
   paradigm = MotorImagery(srate=250)
   X, y, meta = paradigm.get_data(
       dataset, subjects=[1], return_concat=True, label_encode=True, n_jobs=1,
   )
   # Composite IDs remain unique across datasets, subjects, sessions and runs.
   columns = ["dataset", "subject", "session", "run", "trial_id"]
   trial_ids = np.asarray([
       json.dumps(list(row), default=str)
       for row in meta[columns].itertuples(index=False, name=None)
   ])
   # Split by the intended evaluation protocol before exporting separate NPZs.

That example downloads data only when the user executes it. The current
MotorImagery path selects EEG; this extension does not automatically extract
aligned EOG/EMG or image/text features. If such data exist, encode them and supply
their matching trial IDs. Choose subject/session/run holdouts according to the
evaluation question. Disjoint trial IDs alone do not prove independence or
prevent leakage from overlapping windows or a previously fitted encoder.

Explicit training and dynamics
------------------------------

``TransformerTrainer`` implements supervised classification with AdamW,
epoch-level linear warmup and cosine learning-rate decay, gradient clipping,
validation-loss early stopping, and atomic ``best.pt`` / ``last.pt`` checkpoints.
History records train/validation loss and accuracy, the learning rate used for
each epoch, and the mean gradient norm measured before clipping. Returned logits
are passed directly to cross-entropy; do not add softmax before the loss.

.. code-block:: python

   from torch.utils.data import DataLoader
   from metabci.brainda.algorithms.deep_learning import (
       TrainingConfig, TransformerTrainer, load_feature_dataset,
       make_eeg_transformer,
   )

   train = load_feature_dataset("train.npz")
   valid = load_feature_dataset("valid.npz")
   model = make_eeg_transformer(
       train.X.shape[1], n_classes=4, auxiliary_dims=train.auxiliary_dims,
   )
   trainer = TransformerTrainer(model, TrainingConfig(output_dir="runs/experiment"))
   train_loader = DataLoader(train, batch_size=64, shuffle=True)
   valid_loader = DataLoader(valid, batch_size=64)
   # Explicit opt-in: this call actually updates model parameters.
   history = trainer.fit(train_loader, valid_loader)

Constructor/import calls perform no training or file writes. ``compute_loss``
builds a loss graph without an optimizer update; call it in eval mode if batch
normalization buffers must remain untouched. ``evaluate`` runs under no-grad
and restores the preceding model mode. These computations do not prove that
optimization will converge or that classification will be effective.

The default recipe is classification. ``nn.CrossEntropyLoss(reduction="mean")``
can also be supplied directly with class weights; weighted epoch losses are
aggregated by the sum of target weights. A different scalar batch-mean
classification criterion can be passed as ``criterion=...``; it must be an ``nn.Module`` with
JSON-compatible ``get_config()`` and no trainable loss parameters. Put trainable
heads/loss parameters in the model. Users reconstruct custom criterion code
before restoring its saved state. This is not a contrastive pretraining or
arbitrary generative-task trainer.

Checkpoints store model configuration/weights, criterion, optimizer, scheduler,
completed epoch, early-stopping history, main-process random states and explicit
DataLoader/sampler generators. Rebuild the matching model and configuration,
then call ``trainer.load_checkpoint(path, trusted=True)`` for a trusted local
checkpoint before ``fit``. Architecture, criterion and training settings must
match; device and output directory may change. Resume is at completed epochs,
not in the middle of a batch. Worker-local random state and custom
dataset/sampler state are not captured. Use ``num_workers=0`` and reproducible
preprocessing to reproduce main-process data order. The caller must restore the
same data, split IDs, preprocessing and external encoder provenance; the
checkpoint does not fingerprint those inputs. ``fit`` leaves last-epoch
weights; load ``best.pt`` explicitly for best-epoch evaluation. Use
``model.eval()`` and ``torch.no_grad()`` for direct prediction after loading.

Demo commands
-------------

Default commands only inspect randomly initialized networks; they do not fit,
save checkpoints or download a dataset:

.. code-block:: bash

   python demos/eeg_transformer.py
   python demos/eeg_transformer.py --with-auxiliary
   python demos/eeg_transformer.py --preset large
   python demos/eeg_transformer.py --train-data prepared_train.npz --n-classes 4

Actual training requires all three explicit data/output arguments in addition
to ``--train``. Both archives must provide ``pair_ids``; the demo rejects the
same archive path or overlapping train/validation IDs. Channel order,
subject/session independence and feature
availability remain part of the experiment protocol.

.. code-block:: bash

   python demos/eeg_transformer.py --train \
       --train-data train.npz --validation-data valid.npz \
       --output-dir runs/experiment --n-classes 4 --device cuda:0

``--model-config`` accepts a JSON object of constructor overrides; data/CLI
arguments set channel count, class count and auxiliary widths. Training options
include epochs, batch size, learning rate, weight decay, warmup, clipping,
patience and seed. ``--resume`` loads a trusted local checkpoint and also
requires ``--train``; keep its model/training settings identical. The training
demo writes ``history.json`` alongside the two checkpoint files.

Validation scope
----------------

On an Apple Silicon Mac CPU, the four new test modules passed 91 checks using
Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.3, scikit-learn 1.9.1 and skorch 1.4.0.
Both preset demos and the auxiliary-feature demo completed finite forward calls.
No fitting or optimizer steps were run. Reproduce the scoped checks with:

.. code-block:: bash

   python -m pytest tests/test_eeg_transformer.py \
       tests/test_transformer_training.py tests/test_transformer_data.py \
       tests/test_transformer_demo.py

Software checks exercise forward shapes/finiteness, trial-ID alignment, missing
modalities, masked-value invariance, batch separation, gradients without weight
updates, state-dict/checkpoint round trips and skorch initialization/inference.
Tests guard against ``fit`` and optimizer steps. These checks do not validate
real EEG classification accuracy, an actual training/resume run, CUDA, every
supported dependency version, or full-package/platform compatibility.
