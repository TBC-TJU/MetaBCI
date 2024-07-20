import importlib
import os

from metabci.brainda.algorithms.deep_learning.TinySleepNet import TinySleepNet


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict
    fold_idx = 0
    model = TinySleepNet(
        config=config,
        output_dir=os.path.join(model_dir, str(fold_idx)),
        use_rnn=True,
        testing=True,
        use_best=True,
    )
    print(model)