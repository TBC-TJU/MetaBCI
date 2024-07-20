import sys
import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd

from sleepstage import stage_dict
from logger import get_logger

ann2label = { ##改动
    "MCS": 1,
    "VS": 0,
    1: "MCS",
    0: "VS"
}


def time2sec(time):
    return time.hour * 3600 + time.minute * 60 + time.second

def prepare(file):

    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_EEGch = args.select_EEGch
    select_EOGch = args.select_EOGch

    # get data name from array one by one
    logger.info("Loading ...")
    logger.info("Signal file: {}".format(file))
    # logger.info("Annotation file: {}".format(ann_fnames[i]))
    print(file)
    #   pyedflib.EdfReader to get file
    psg_f = pyedflib.EdfReader(file)
    # ann_f = pd.read_excel(ann_fnames[i],sheet_name="Sheet2")##选择工作簿  改动

    #   test getStartdatetime getFileDuration datarecord_duration
    # assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
    start_datetime = psg_f.getStartdatetime()
    logger.info("Start datetime: {}".format(str(start_datetime)))

    file_duration = psg_f.getFileDuration()
    logger.info("File duration: {} sec".format(file_duration))
    epoch_duration = psg_f.datarecord_duration

    if psg_f.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        epoch_duration = epoch_duration / 2
        logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
    else:
        logger.info("Epoch duration: {} sec".format(epoch_duration))

    # Extract signal from the selected channel
    ch_names = psg_f.getSignalLabels()
    ch_samples = psg_f.getNSamples()
    select_EEGch_idx = -1
    select_EOGch_idx = -1

    for s in range(psg_f.signals_in_file):
        if ch_names[s] == select_EEGch:
            select_EEGch_idx = s
            break
    for s in range(psg_f.signals_in_file):
        if ch_names[s] == select_EOGch or "EOG2" in ch_names[s]:
            select_EOGch_idx = s
            break
    if select_EEGch_idx == -1:
        raise Exception("EEG Channel not found.")
    if select_EOGch_idx == -1:
        raise Exception("EOG Channel not found.")

    # getSampleFrequency !!!
    sampling_rate = psg_f.getSampleFrequency(select_EEGch_idx)
    epoch_duration = 30  # 改动
    eeg = psg_f.readSignal(select_EEGch_idx)
    compute=len(eeg)%3000 #3000
    eeg = eeg if compute==0 else eeg[:len(eeg)-compute]
    # print(eeg.shape)
    n_epoch_samples = int(epoch_duration * sampling_rate)
    EEGsignals = np.reshape(eeg,(-1, n_epoch_samples)) ##修改维度
    print(EEGsignals.shape)

    # getSampleFrequency !!!
    sampling_rate = psg_f.getSampleFrequency(select_EOGch_idx)
    n_epoch_samples = int(epoch_duration * sampling_rate) ##修改
    eog = psg_f.readSignal(select_EOGch_idx)
    compute = len(eog) % 3000 #3000
    eog = eog if compute == 0 else eog[:len(eog) - compute]
    print(eog.shape)
    EOGsignals = np.reshape(eog,(-1, n_epoch_samples)) ##修改维度
    print(EOGsignals.shape)
    # logger.info("Select channel: {}".format(select_EOGch))
    # logger.info("Select channel samples: {}".format(ch_samples[select_EOGch_idx]))
    # logger.info("Sample rate: {}".format(sampling_rate))

    # Sanity check
    n_epochs = psg_f.datarecords_in_file
    print(n_epochs)
    n_epochs=len(EEGsignals)#改动
    print(len(EEGsignals))
    if psg_f.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
        n_epochs = n_epochs * 2
    assert len(EEGsignals) == n_epochs

    # Generate labels from onset and duration annotation
    # labels = []
    labels = np.ones(len(EEGsignals))

    jarPath=os.environ.get("SLEEPNET_JAR_PATH")
    file= file.split("/")[-1]
    sleepPreFile = jarPath+"/sleepstage/out_sleepedf/predict/pred_"+file.split(".")[0]+".npz"
    data1 = np.load(sleepPreFile, allow_pickle=True)
    sleepstages = np.ones((len(data1['z_pred']), 3000), dtype=int)
    j = 0
    for i in data1['z_pred']:
        sleepstages[j] = sleepstages[j] * i
        j = j + 1

    # Get epochs and their corresponding labels
    x = EEGsignals.astype(np.float32)
    y = EOGsignals.astype(np.float32)
    s = sleepstages.astype(np.float32)
    z = labels.astype(np.int32)   ##标签

    logger.info("Data after removal: {}, {}, {}".format(x.shape, y.shape, z.shape))


    # Save
    filename = ntpath.basename(file.split('/')[-1]).replace(".edf", ".npz")
    print(filename)
    save_dict = {
        "x": x,
        "y": y,
        "s": s,
        "z": z,
        "fs": sampling_rate,
        "EEGch_label": select_EEGch,
        "EOGch_label": select_EOGch,
        "start_datetime": start_datetime,
        "file_duration": file_duration,
        "epoch_duration": epoch_duration,
        "n_all_epochs": n_epochs,
        "n_epochs": len(x),
    }
    np.savez(os.path.join(args.output_dir, filename), **save_dict)

    logger.info("\n=======================================\n")


if __name__ == "__main__":
    path=os.environ.get("SLEEPNET_JAR_PATH")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=path+"/DOC_valuation/data/sleepedf/sleep-cassette/eeg_eog/",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_EEGch", type=str, default="C3",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--select_EOGch", type=str, default="HEOG",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    prepare(
        file=args.file,
    )
