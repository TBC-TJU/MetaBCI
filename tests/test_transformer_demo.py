"""Keep demo defaults free of training, downloads and output artifacts."""

import json
import sys

import pytest
import torch

from demos import eeg_transformer
from metabci.brainda.algorithms.deep_learning import TransformerTrainer


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    previous_threads = torch.get_num_threads()

    def forbidden(*args, **kwargs):
        pytest.fail("The default demo must not train")

    monkeypatch.setattr(TransformerTrainer, "fit", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    yield
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("options", [[], ["--with-auxiliary"]])
def test_default_demo_only_inspects_untrained_model(
    monkeypatch, tmp_path, capsys, options
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eeg_transformer.py", *options])
    eeg_transformer.main()
    report = json.loads(capsys.readouterr().out)
    assert report["training_executed"] is False
    assert report["finite"] is True
    assert report["input_shape"] == [4, 22, 1000]
    assert report["feature_shape"] == [4, 64]
    assert report["logit_shape"] == [4, 4]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "options",
    [
        ["--train"],
        ["--resume", "checkpoint.pt"],
        [
            "--train",
            "--train-data",
            "same.npz",
            "--validation-data",
            "same.npz",
            "--output-dir",
            "output",
        ],
    ],
)
def test_training_and_resume_require_explicit_arguments(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["eeg_transformer.py", *options])
    with pytest.raises(SystemExit) as caught:
        eeg_transformer._arguments()
    assert caught.value.code == 2


def test_validation_overlap_and_auxiliary_width_are_rejected():
    X = torch.zeros(2, 4, 100)
    reference = eeg_transformer.EEGFeatureDataset(X, [0, 1], pair_ids=["a", "b"])
    heldout = eeg_transformer.EEGFeatureDataset(X, [0, 1], pair_ids=["a", "c"])
    model = eeg_transformer.make_eeg_transformer(4, 2)
    with pytest.raises(ValueError, match="disjoint"):
        eeg_transformer._validate_dataset(heldout, 2, model, reference)
    mismatched = eeg_transformer.EEGFeatureDataset(
        X,
        [0, 1],
        auxiliary={"context": torch.zeros(2, 6)},
        pair_ids=["a", "b"],
        auxiliary_pair_ids={"context": ["a", "b"]},
    )
    with pytest.raises(ValueError, match="width/name"):
        eeg_transformer._validate_dataset(mismatched, 2, model)
