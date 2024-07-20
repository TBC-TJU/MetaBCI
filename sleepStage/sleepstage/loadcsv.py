import argparse
import glob
import importlib
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import shutil
import datetime
import sklearn.metrics as skmetrics
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import csv
import pandas as pd
from models import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from matplotlib import pyplot as plt
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations

path=os.environ.get("SLEEPNET_JAR_PATH")
# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1
N2 = 2      # Stage N2
N3 = 3      # Stage N3
REM = 4     # Stage REM
MOVE = 5    # Movement
UNK = 6     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}
class_dict = {
    # W: "W",
    # N1: "N1",
    # N2: "N2",
    # N3: "N3",
    # REM: "REM",
    # MOVE: "MOVE",
    # UNK: "UNK",
    W: "W期",
    N1: "N1期",
    N2: "N2期",
    N3: "N3期",
    REM: "REM期",
    MOVE: "Movement",
    UNK: "未知",
}

def loadData(j):
    current_time = datetime.datetime.now()
    a = "x"
    b = "y"
    aa = []
    path = os.environ.get("SLEEPNET_JAR_PATH")
    data = np.loadtxt('./test.csv')
    # print(data)
    data = data.T
    # print(data)
    params = BrainFlowInputParams()
    board = BoardShim(532, params)
    eeg_channels = board.get_eeg_channels(532)
    sampling_rate = board.get_sampling_rate(532)
    # print(sampling_rate)
    # eeg_channels = [1, 2,3,4]
    o = data[eeg_channels]
    for count, channel in enumerate(eeg_channels):
        # plot timeseries
        DataFilter.detrend(o[channel - 1], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(o[channel - 1], sampling_rate, 3.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(o[channel - 1], sampling_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(o[channel - 1], sampling_rate, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
    # print(o)

    length = len(o[0])
    if(length - j < 15000):
        if(length - j == 0):
            j = 0
        bb = o[2][j:j+(length-j)//3000*3000]
    else:
        bb = o[2][j:j+(length-j)//3000*3000]
    # bb = o[j:j + 15000]
    # print("bb的长度")
    # print(len(bb))
    # if len(bb) != 15000:
    #     X1 = []
    #     X2 = []
    #     X3 = []
    #     return X1, X2, X3, current_time.strftime('%Y-%m-%d %H:%M:%S')
    # j = j + 15000 + 3000

    aa = o[0][j:j+(length-j)//3000*3000]-o[1][j:j+(length-j)//3000*3000] #EOG
    # aa = list(int(i) for i in o)
    # print(len(aa))

    X1 = bb
    X2 = aa
    length = len(X1)//3000
    X1 = np.array(X1)
    X1 = X1.reshape(length, 3000)
    X2 = np.array(X2)
    X2 = X2.reshape(length, 3000)
    X3 = np.ones((length))

    x = np.squeeze(X1)
    x = x[:, :, np.newaxis, np.newaxis]

    y = np.squeeze(X2)
    y = y[:, :, np.newaxis, np.newaxis]

    # Casting
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = X3.astype(np.int32)
    EEGsignals = []
    EOGsignals = []
    labels = []
    EEGsignals.append(x)
    EOGsignals.append(y)
    labels.append(z)

    return EEGsignals, EOGsignals, labels, current_time.strftime('%Y-%m-%d %H:%M:%S')

def predict(
    begin,
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

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    test_x, test_y, test_z, start_datetime = loadData(begin)
    if len(test_x) == 0:
        # print("最后一个不到15000，分期结束")
        return

    print_n_samples_each_class(np.hstack(test_z))

    if config["model"] == "model-origin":
        for night_idx, night_data in enumerate(zip(test_x, test_y, test_z)):
            # Create minibatches for testing
            night_x, night_y, night_z = night_data
            test_minibatch_fn = iterate_batch_seq_minibatches(
                night_x,
                night_y,
                night_z,
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
            )
            # Evaluate
            test_outs = model.evaluate(test_minibatch_fn)
            s_preds.extend(test_outs["test/preds"])
            preds.extend(test_outs["test/preds"])
            return test_outs["test/preds"],start_datetime
    else:
        for night_idx, night_data in enumerate(zip(test_x, test_y, test_z)):
            # Create minibatches for testing
            night_x, night_y, night_z = night_data
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x],
                [night_y],
                [night_z],
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
            s_preds.extend(test_outs["test/preds"])
            preds.extend(test_outs["test/preds"])
            return test_outs["test/preds"],start_datetime

        tf.reset_default_graph()

# 读取预测predict后的npz
def stageOfNpz(
    begin,
    config_file=path+"\sleepstage\config\sleepedf.py",
    model_dir=path+"\sleepstage\out_sleepedf\train",
    output_dir=path+"\sleepstage\out_sleepedf\predict",
    log_file=path+"\sleepstage\out_sleepedf\predict.log",
    use_best=True,
):
    data,start_datetime = predict(
        begin=begin,
        config_file=config_file,
        model_dir=model_dir,
        output_dir=output_dir,
        log_file=log_file,
        use_best=use_best,
    )
    startTime = start_datetime
    time = datetime.datetime.strptime(startTime, "%Y-%m-%d %H:%M:%S")
    offset1 = datetime.timedelta(seconds=-180)
    time = time + offset1
    # 计算偏移量
    offset = datetime.timedelta(seconds=+1)
    # 获取修改后的时间并格式化
    length=len(data)
    if begin>length:
        begin=length-500
    if begin < 0:
        begin = 0
    n = begin + 500
    if n>length:
        n=length
    result = ""
    for i in np.arange(begin,n):
        if i == n - 1:
            s = str((time + offset).strftime("%Y-%m-%d %H:%M:%S"))
            result += ("%s" % s)+" "+("%s" % class_dict[data[i]])
            break
        else:
            s = str((time + offset).strftime("%Y-%m-%d %H:%M:%S"))
            result += ("%s" % s)+" "+("%s," % class_dict[data[i]])
            time = (time + offset)
    print(result)
    return result


if __name__ == "__main__":
    path=os.environ.get("SLEEPNET_JAR_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin", type=int, default=0, required=False)
    parser.add_argument("--config_file", type=str, required=False,
                        default=path+"\sleepNet\config\sleepedf.py")
    parser.add_argument("--model_dir", type=str, required=False,
                        default=path+"\sleepNet\out_sleepedf\train")
    parser.add_argument("--output_dir", type=str, required=False,
                        default=path+"\sleepNet\out_sleepedf\predict")
    parser.add_argument("--log_file", type=str, required=False,
                        default=path+"\sleepNet\out_sleepedf\predict.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    stageOfNpz(
        begin=args.begin,
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_best=args.use_best,
    )