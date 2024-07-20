import os
import re

import numpy as np


def get_subject_files(dataset, files, sid):
    """Get a list of files storing each subject data."""
    # Pattern of the subject files from different datasets
    if "mass" in dataset:
        reg_exp = f".*-00{str(sid + 1).zfill(2)} PSG.npz"
        # reg_exp = "SS3_00{}\.npz$".format(str(sid+1).zfill(2))
    elif "sleepedfx" in dataset:
        reg_exp = fr'{str(sid + 1).zfill(2)}-\d{{2}}-[a-zA-Z\s]+ PSG\.npz'
    elif "sleepedf" in dataset:
        # reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
        # reg_exp = "[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(str(sid).zfill(2))
        reg_exp = f".+{str(sid).zfill(2)}\.npz$"
    elif "isruc" in dataset:
        reg_exp = f"subject{sid + 1}.npz"
    else:
        raise Exception("Invalid datasets.")

    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
    return subject_files


def load_data(subject_files):
    """Load data from subject files."""

    EEGsignals = []
    EOGsignals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            z = f['z']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]

            y = np.squeeze(y)
            y = y[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            z = z.astype(np.int32)

            EEGsignals.append(x)
            EOGsignals.append(y)
            labels.append(z)

    return EEGsignals, EOGsignals, labels, sampling_rate
