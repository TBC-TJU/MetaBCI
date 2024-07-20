params = {
     # Train
    "n_epochs": 200,
    "learning_rate": 5e-5,  # 原来：1e-4
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "clip_grad_value": 5.0,  # 5.0
    "evaluate_span": 50,
    "checkpoint_span": 50,
    "finetune_model_dir": r"./out_sleepedf_v6/finetune",

    # Early-stopping
    "no_improve_epochs": 100,

    # Model
    "model": "model-mod-8",
    "n_rnn_layers": 1,
    "n_rnn_units": 128,
    "sampling_rate": 100.0,
    "input_size": 3000,
    "n_classes": 5,#改动
    "l2_weight_decay": 1e-3,

    # Dataset
    "dataset": "sleepedf",
    "data_dir": "./data/sleepedf/sleep-cassette/eeg_eog",
    "n_folds": 15,
    "n_subjects": 15,

    # Data Augmentation
    "augment_seq": True,
    "augment_signal_full": True,
    "weighted_cross_ent": True,
}

train = params.copy()
train.update({
    "seq_length": 20,
    "batch_size": 15,
})

predict = params.copy()
predict.update({
    "training_mode": 'predict',
    "batch_size": 1,
    "seq_length": 1,
})

finetune = params.copy()
finetune.update({
    # train
    "training_mode": 'finetune',
    "finetune_model_dir": r"./out_sleepedf_v6/finetune",
    "seq_length": 20,
    "batch_size": 15,
})