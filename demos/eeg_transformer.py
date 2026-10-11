"""Inspect an untrained EEG Transformer; training requires explicit opt-in.

Run from a checkout installed with MetaBCI's deep-learning dependencies. By
default this uses synthetic tensors, performs only eval/no-grad forward calls,
and writes no files. Real data are supplied as documented NPZ archives.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from metabci.brainda.algorithms.deep_learning import (
    EEGFeatureDataset,
    TrainingConfig,
    TransformerTrainer,
    load_feature_dataset,
    make_eeg_transformer,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("small", "large"), default="small")
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--model-config", type=Path, help="JSON constructor overrides")
    parser.add_argument("--train-data", type=Path, help="NPZ training/input archive")
    parser.add_argument("--validation-data", type=Path, help="Separate validation NPZ")
    parser.add_argument(
        "--with-auxiliary", action="store_true", help="Synthetic demo only"
    )
    parser.add_argument(
        "--train", action="store_true", help="Explicitly enable training"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Trusted local training checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--device", default="cpu", help="cpu or an available cuda device"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if args.n_classes < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("n-classes, batch-size and cpu-threads must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must lie in [0, 2**32)")
    if args.train and not all((args.train_data, args.validation_data, args.output_dir)):
        parser.error(
            "--train requires --train-data, --validation-data and --output-dir"
        )
    if args.train and args.train_data.resolve() == args.validation_data.resolve():
        parser.error("Training and validation must use separate archive paths")
    if args.resume and not args.train:
        parser.error("--resume requires explicit --train")
    if args.with_auxiliary and args.train_data:
        parser.error("--with-auxiliary is only for the synthetic forward demo")
    return args


def _synthetic_dataset(n_classes, seed, with_auxiliary):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((4, 22, 1000)).astype(np.float32)
    y = np.arange(4, dtype=np.int64) % n_classes
    if not with_auxiliary:
        return EEGFeatureDataset(X, y)
    # Auxiliary rows deliberately arrive in a different order. The dataset
    # joins them by unique trial IDs, never by label or accidental row order.
    return EEGFeatureDataset(
        X,
        y,
        auxiliary={"context": rng.standard_normal((3, 2, 8)).astype(np.float32)},
        pair_ids=["demo-0", "demo-1", "demo-2", "demo-3"],
        auxiliary_pair_ids={"context": ["demo-2", "demo-0", "demo-1"]},
    )


def _validate_dataset(dataset, n_classes, model, reference=None):
    if int(dataset.y.max()) >= n_classes:
        raise ValueError("Class labels must be integers in [0, n_classes)")
    if dataset.X.shape[1] != model.n_channels:
        raise ValueError("EEG channel count differs between model and dataset")
    if dataset.X.shape[2] < model.minimum_samples:
        raise ValueError("EEG windows are shorter than the configured stem supports")
    for name, width in dataset.auxiliary_dims.items():
        if model.auxiliary_dims.get(name) != width:
            raise ValueError("Auxiliary feature width/name differs: " + name)
    if reference is not None and reference.pair_ids and dataset.pair_ids:
        if set(reference.pair_ids).intersection(dataset.pair_ids):
            raise ValueError("Training and validation trial IDs must be disjoint")


def main():
    args = _arguments()
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = (
        load_feature_dataset(args.train_data)
        if args.train_data
        else _synthetic_dataset(args.n_classes, args.seed, args.with_auxiliary)
    )
    validation = None
    if args.train:
        validation = load_feature_dataset(args.validation_data)
        if dataset.pair_ids is None or validation.pair_ids is None:
            raise ValueError("Training demo requires pair_ids in both data archives")
    overrides = {}
    if args.model_config:
        with args.model_config.open(encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError("model-config must be a JSON object")
        if {"n_channels", "n_classes", "auxiliary_dims"}.intersection(overrides):
            raise ValueError(
                "Channels/classes/modalities are set by data and CLI arguments"
            )
    model = make_eeg_transformer(
        dataset.X.shape[1],
        args.n_classes,
        args.preset,
        auxiliary_dims=dataset.auxiliary_dims,
        **overrides,
    )
    _validate_dataset(dataset, args.n_classes, model)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        clip_grad_norm=args.grad_clip,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        output_dir=str(args.output_dir) if args.output_dir else None,
    )
    trainer = TransformerTrainer(model, config)
    model.eval()
    batch = next(iter(DataLoader(dataset, batch_size=min(4, len(dataset)))))
    with torch.no_grad():
        logits = trainer.forward_batch(batch)
        features = model.cal_backbone(
            batch["X"].to(trainer.device),
            auxiliary={
                k: v.to(trainer.device) for k, v in batch.get("auxiliary", {}).items()
            },
            auxiliary_mask={
                k: v.to(trainer.device)
                for k, v in batch.get("auxiliary_mask", {}).items()
            },
        )
    print(
        json.dumps(
            {
                "preset": args.preset,
                "parameters": sum(p.numel() for p in model.parameters()),
                "model_config": model.get_config(),
                "input_shape": list(batch["X"].shape),
                "feature_shape": list(features.shape),
                "logit_shape": list(logits.shape),
                "finite": bool(torch.isfinite(logits).all()),
                "training_executed": False,
            },
            indent=2,
        )
    )
    if not args.train:
        return
    _validate_dataset(validation, args.n_classes, model, reference=dataset)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    if args.resume:
        trainer.load_checkpoint(args.resume, trusted=True)
    history = trainer.fit(train_loader, valid_loader)
    history_path = args.output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(
        json.dumps(
            {
                "training_executed": True,
                "epochs_completed": trainer.epoch,
                "history": str(history_path),
            }
        )
    )


if __name__ == "__main__":
    main()
