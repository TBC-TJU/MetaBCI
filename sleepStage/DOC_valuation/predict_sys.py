import sys
import argparse
import glob
import importlib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import sklearn.metrics as skmetrics
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from data import load_data, get_subject_files
from models import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger

def predict(
    file,
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
):
    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create logger
    logger = get_logger(log_file, level="info")

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    trues = []
    preds = []
    fold_idx=0

    model = TinySleepNet(
        config=config,
        output_dir=os.path.join(model_dir, str(fold_idx)),
        use_rnn=True,
        testing=True,
        use_best=use_best,
    )

    # Get corresponding files
    s_trues = []
    s_preds = []

    test_files = []
    test_files.append(file)

    # for vf in test_files: logger.info("Load files {} ...".format(vf))
    dataTemp =  np.load(file,allow_pickle=True)
    start_datetime = dataTemp['start_datetime']
    file_duration = dataTemp['file_duration']

    test_x, test_y, test_z, test_s, _ = load_data(test_files) # 新增睡眠分期结果

    print_n_samples_each_class(np.hstack(test_z))
    if config["model"] == "model-origin":
        for night_idx, night_data in enumerate(zip(test_x, test_y, test_z, test_s)):
            # Create minibatches for testing
            night_x, night_y, night_z, night_s = night_data  # 新增睡眠分期结果
            test_minibatch_fn = iterate_batch_seq_minibatches(
                night_x,
                night_y,
                night_z,
                sleepinputs=night_s,  # 新增睡眠分期结果
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
            )
            # Evaluate
            test_outs = model.evaluate(test_minibatch_fn)
            s_trues.extend(test_outs["test/trues"])
            s_preds.extend(test_outs["test/preds"])
            trues.extend(test_outs["test/trues"])
            preds.extend(test_outs["test/preds"])

            # Save labels and predictions (each night of each subject)
            save_dict = {
                "z_true": test_outs["test/trues"],
                "z_pred": test_outs["test/preds"],
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "pred_{}.npz".format(fname)
            )
            np.savez(save_path, **save_dict)
    else:
        for night_idx, night_data in enumerate(zip(test_x, test_y, test_z, test_s)):
            # Create minibatches for testing
            night_x, night_y, night_z, night_s = night_data
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                eeginputs=[night_x],
                eoginputs=[night_y],
                targets=[night_z],
                sleepinputs=[night_s],  # 新增睡眠分期结果
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            if (config.get('augment_signal') is not None) and config['augment_signal']:
                # Evaluate
                test_outs = model.evaluate_aug(test_minibatch_fn)
            else:
                # Evaluate
                test_outs = model.evaluate(test_minibatch_fn)
            s_trues.extend(test_outs["test/trues"])
            s_preds.extend(test_outs["test/preds"])
            trues.extend(test_outs["test/trues"])
            preds.extend(test_outs["test/preds"])
            # Save labels and predictions (each night of each subject)
            pred = sum(test_outs["test/preds"])/len(test_outs["test/preds"])
            save_dict = {
                "start_datetime": start_datetime,
                "file_duration": file_duration,
                "z_true": test_outs["test/trues"],
                "z_pred": test_outs["test/preds"],
                "pred": pred
            }
            fname = os.path.basename(test_files[night_idx]).split(".")[0]
            save_path = os.path.join(
                output_dir,
                "pred_{}.npz".format(fname)
            )
            # print(pred)
            if pred > 0.5:
                print("1")  # MCS
            else:
                print("0")  # VS
            # print(sum(test_outs["test/preds"])/len(test_outs["test/preds"]))
            np.savez(save_path, **save_dict)
            # logger.info("Saved outputs to {}".format(save_path))

    tf.reset_default_graph()

if __name__ == "__main__":
    path=os.environ.get("SLEEPNET_JAR_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=False)
    parser.add_argument("--config_file", type=str, required=False, default=path+"/DOC_valuation/config/sleepedf.py")
    parser.add_argument("--model_dir", type=str, required=False, default=path+"/DOC_valuation/out_sleepedf/train")
    parser.add_argument("--output_dir", type=str, required=False, default=path+"/DOC_valuation/out_sleepedf/predict")
    parser.add_argument("--log_file", type=str, required=False, default=path+"/DOC_valuation/out_sleepedf/predict.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    predict(
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        file=args.file,
        use_best=args.use_best,
    )
