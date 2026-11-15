#!/usr/bin/env python3
"""Offline OVO parity check on local BNCI2014-001 subjects 1/2/3.

Example: python demos/csp_ovo_validation.py --data-dir /path/to/mat --output-dir /tmp/ovo
Supply A01E.mat/A01T.mat, A02E.mat/A02T.mat and A03E.mat/A03T.mat; no downloads occur.
For the old implementation, use --baseline-source /path/to/original-csp.py in sklearn 1.5.
All source trials, including artifact-flagged trials, are retained. This checks a
compatibility repair; it does not measure online performance or accuracy improvement.
"""

# ruff: noqa: E402
import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[name] = "1"

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import mne
import numpy as np
import scipy
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from metabci.brainda.algorithms.decomposition import csp
from metabci.brainda.datasets.bnci import BNCI2014001
from metabci.brainda.paradigms import MotorImagery


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalBNCI2014001(BNCI2014001):
    """Override only file resolution; preserve the native [E], [T] session order."""

    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    def data_path(self, subject, *args, **kwargs):
        if subject not in self.subjects:
            raise ValueError(f"Invalid subject: {subject}")
        files = [self.data_dir / f"A{subject:02d}{suffix}.mat" for suffix in ("E", "T")]
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(
                    f"A complete local source file is required: {path}"
                )
        return [[str(path)] for path in files]


def implementation(baseline_source):
    if baseline_source is None:
        return csp
    name = "metabci.brainda.algorithms.decomposition.csp_before"
    spec = importlib.util.spec_from_file_location(name, baseline_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CSP source: {baseline_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def epochs(dataset, subject):
    paradigm = MotorImagery(
        channels=dataset.channels,
        events=["left_hand", "right_hand", "feet", "tongue"],
        intervals=[(2, 6)],
    )

    def raw_hook(raw, caches):
        raw.filter(
            8,
            30,
            picks=dataset.channels,
            method="iir",
            phase="zero",
            n_jobs=1,
            iir_params={"order": 4, "ftype": "butter", "output": "sos"},
            verbose=False,
        )
        return raw, caches

    # Every native Raw is a separate source run; filtering never crosses runs/days.
    paradigm.register_raw_hook(raw_hook)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=[subject],
        label_encode=False,
        return_concat=True,
        n_jobs=1,
        verbose=False,
    )
    meta = meta.assign(_run_number=meta["run"].str.removeprefix("run_").astype(int))
    order = meta.sort_values(
        ["session", "_run_number", "trial_id"], kind="stable"
    ).index.to_numpy()
    X = np.ascontiguousarray(X[order], dtype=np.float64)
    y = np.asarray(y[order], dtype=np.int64)
    meta = meta.iloc[order].drop(columns="_run_number").reset_index(drop=True)
    if X.shape != (576, 22, 1000) or not np.isfinite(X).all():
        raise ValueError(f"Expected finite (576, 22, 1000) EEG epochs; got {X.shape}")
    return X, y, meta


def evaluate(dataset, subject, module, output_dir):
    sources = {
        suffix: dataset.data_dir / f"A{subject:02d}{suffix}.mat"
        for suffix in ("T", "E")
    }
    source_hashes = {suffix: sha256(path) for suffix, path in sources.items()}
    X, y, meta = epochs(dataset, subject)
    train = np.flatnonzero(meta["session"].eq("session_1").to_numpy())
    test = np.flatnonzero(meta["session"].eq("session_0").to_numpy())
    for selected in (train, test):
        labels, counts = np.unique(y[selected], return_counts=True)
        if (
            len(selected) != 288
            or meta.iloc[selected]["run"].nunique() != 6
            or not np.array_equal(labels, [1, 2, 3, 4])
            or not np.array_equal(counts, [72, 72, 72, 72])
        ):
            raise ValueError(
                "Expected 288 trials, six runs and 72 trials/class per day"
            )
    np.random.seed(42)
    start = time.perf_counter()
    model = module.MultiCSP(n_components=2, multiclass="ovo")
    model.fit(X[train], y[train])
    train_features, test_features = model.transform(X[train]), model.transform(X[test])
    classifier = SVC().fit(train_features, y[train])
    prediction = classifier.predict(test_features)
    decision = classifier.decision_function(test_features)
    elapsed = time.perf_counter() - start
    arrays = {
        "train_features": train_features,
        "test_features": test_features,
        "prediction": prediction,
        "decision_function": decision,
        "y_train": y[train],
        "y_test": y[test],
        "train_indices": train,
        "test_indices": test,
        "classifier_classes": classifier.classes_,
    }
    for column in meta:
        values = meta[column].to_numpy()
        values = values.astype(str) if values.dtype.kind == "O" else values
        arrays[f"meta_{column}_train"], arrays[f"meta_{column}_test"] = (
            values[train],
            values[test],
        )
    prefix = output_dir / f"sub-{subject:02d}-ovo"
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    report = {
        "subject": subject,
        "strategy": "ovo",
        "n_train": 288,
        "n_test": 288,
        "train": "T / session_1",
        "heldout": "E / session_0; different recording day",
        "metrics": {
            "accuracy": float(accuracy_score(y[test], prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
            "cohen_kappa": float(cohen_kappa_score(y[test], prediction)),
            "confusion_matrix": confusion_matrix(
                y[test], prediction, labels=[1, 2, 3, 4]
            ).tolist(),
            "confusion_labels": [1, 2, 3, 4],
        },
        "parameters": {
            "bandpass": [8, 30],
            "iir_order": 4,
            "ftype": "butter",
            "output": "sos",
            "phase": "zero",
            "interval_half_open": [2, 6],
            "channels": dataset.channels,
            "sfreq": 250,
            "n_components": 2,
            "classifier": "SVC()",
            "seed": 42,
            "artifact_policy": "retain all source trials",
            "tuning": "none",
            "cpu_thread_limit": 1,
            "joblib_backend": "threading",
        },
        "versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "mne": mne.__version__,
            "python": sys.version,
        },
        "source_mat_sha256": source_hashes,
        "implementation_sha256": sha256(Path(module.__file__)),
        "seconds": elapsed,
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"subject": subject, "metrics": report["metrics"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    module = implementation(args.baseline_source)
    dataset = LocalBNCI2014001(args.data_dir.resolve())
    with threadpool_limits(limits=1), joblib.parallel_backend("threading", n_jobs=1):
        for subject in args.subjects:
            evaluate(dataset, subject, module, args.output_dir)


if __name__ == "__main__":
    main()
