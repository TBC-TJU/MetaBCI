# Local macOS validation of the MultiCSP OVO adapter

The adapter restores OVO fitting with current scikit-learn while preserving the
original three-dimensional CSP calculation. These are local validation results
for the patched package, with an offline, fixed-protocol real-EEG parity check.
See the [mapping proof and its assumptions](csp-ovo-proof.md) and the
[machine-readable results, source hashes and quality-check counts](csp-ovo-mac.json).

## Environment and package checks

Apple M1 Pro, arm64, 32 GiB RAM, macOS 27.0.1; Python 3.12.14.
Shared numerical/package versions: NumPy 2.5.3, SciPy 1.18.1, MNE 1.13.2,
PyTorch 2.14.1, and joblib 1.6.0. CSP calculations use the CPU.

| scikit-learn | Algorithm and regression tests |
| --- | --- |
| 1.5.2 | 178 passed |
| 1.6.1 | 178 passed |
| 1.9.1 | 178 passed |

This includes the existing algorithm suite and 11 adapter regressions, including
automatic component selection, OVR reference features, process-backend compatibility,
outer pipeline cross-validation, and the FBMultiCSP OVO filter-bank path.
Two existing `ComplexWarning` messages came from the original grosse-wentrup path.
All new/changed files passed pre-commit (including Ruff 0.16.4). Repository-wide
checks still report existing diagnostics. An archived unpatched base at
`3bc9643f3c39a616e73fa4872e6a77c536ad6107`, checked with the same tools, gave
Ruff errors 36 → 35, files needing formatting 13 → 12, and mypy 2.4.0 errors
38 → 38; no new diagnostics were introduced. These are local checks, not a
claim that GitHub CI has passed.
`pip check` passed. Seven import-smoke entries passed: `metabci`, brainda datasets,
brainda paradigms, decomposition CSP, brainflow amplifiers, brainflow workers,
and `pylsl` (native libLSL `library_version() == 118`).
Import success does not establish stimulus presentation or device/stream timing.

## Real EEG protocol

Data source: [BNCI Horizon 2020, four-class motor imagery 001-2014](https://bnci-horizon-2020.eu/database/data-sets),
originally BCI Competition IV dataset 2a; see the [official description](https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf).
Subjects 1, 2, and 3 were selected before evaluation; no other subjects were screened.
The source files are `A01T.mat`, `A01E.mat`, `A02T.mat`, `A02E.mat`, `A03T.mat`, and `A03E.mat`.
The local-file subclass changes file resolution only; native MetaBCI loading and
MotorImagery epoch extraction are used. Native file ordering remains `[E, T]`:
`session_0` is E/heldout and `session_1` is T/training, on different recording days.

Each original run is filtered separately at 8–30 Hz with a fourth-order Butterworth
IIR, SOS representation and zero phase, using the 22 EEG channels; runs/days are not merged for filtering.
Epochs use the half-open interval `[2, 6)` seconds from trial start: 1,000 points at 250 Hz.
Each subject has 288 training and 288 heldout trials, six runs per day and 72 trials per class.
Every source trial is retained, including artifact flags; EOG channels are not classifier inputs.
`MultiCSP(n_components=2, multiclass="ovo")` gives 12 features, followed by an outer `SVC()`.
There is no hyperparameter tuning; seed 42 is set. CSP and SVC fit only the T day.
The E day is used only for transformation and evaluation. Native threads and joblib
are bounded to one CPU worker for this real-EEG comparison.

| Subject | Accuracy, all retained E trials | Flagged T trials retained | Flagged E trials retained |
| --- | --- | --- | --- |
| 1 | 58.68% | 15 | 7 |
| 2 | 46.88% | 18 | 5 |
| 3 | 63.89% | 18 | 15 |
| Mean | 56.48% | — | — |

The original implementation under scikit-learn 1.5.2 is the baseline.
For patched 1.5.2, 1.6.1 and 1.9.1, every subject's training features, heldout
features and outer decision values had maximum absolute difference **0** from
that baseline; heldout prediction changes were **0** across **864 trials per version**.
Source hashes, labels, trial indices and metadata were matched for the paired comparison.
This is observed parity on these runs, not a universal guarantee of bitwise equivalence.

## Reproduction

From the repository root, prepare an isolated Python 3.12 environment:

```sh
python3.12 -m venv .venv-ovo
.venv-ovo/bin/python -m pip install -e '.[brainda,brainflow,dev]' 'numpy==2.5.3' 'scipy==1.18.1' 'mne==1.13.2' 'torch==2.14.1' 'joblib==1.6.0' 'scikit-learn==1.9.1'
.venv-ovo/bin/python -m pip check
LOKY_MAX_CPU_COUNT=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv-ovo/bin/python -m pytest -q -o addopts= tests
```

Place the six official MAT files in `data-bnci001/`; the demo does not download data.
Run the patched implementation and save local features, predictions and manifests:

```sh
.venv-ovo/bin/python demos/csp_ovo_validation.py --data-dir data-bnci001 --output-dir output-1.9.1 --subjects 1 2 3
git show 3bc9643f3c39a616e73fa4872e6a77c536ad6107:metabci/brainda/algorithms/decomposition/csp.py > original-csp.py
```

Repeat with independent environments pinned to scikit-learn 1.5.2 and 1.6.1.
In the 1.5.2 environment, run the same demo additionally with
`--baseline-source original-csp.py --output-dir output-baseline-1.5.2`.
Compare each subject's saved `train_features`, `test_features`, `decision_function`,
`prediction`, labels and indices against the baseline; also match MAT hashes in the manifests.
The demo emits JSON manifests and NPZ outputs locally. No raw EEG files or trial arrays
are committed with this report.

## Scope

The repair does not demonstrate an accuracy improvement. The reported accuracy retains
artifact-flagged trials and is not the competition's artifact-free scoring procedure;
zero-phase filtering also makes this an offline, non-causal evaluation.
This validates the stated Mac/package/OVO paths, not complete cross-platform support,
all MetaBCI features, brainstim, EEG hardware, online decoding or real-time latency.
