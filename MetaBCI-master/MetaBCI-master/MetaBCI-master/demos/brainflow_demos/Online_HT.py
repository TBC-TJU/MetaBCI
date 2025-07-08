# -*- coding: utf-8 -*-
"""
SSVEP Feedback on HTOnlineSystem.

"""
import time
import numpy as np

from metabci.brainflow.workers import ProcessWorker
from metabci.brainflow.amplifiers import HTOnlineSystem, Marker

class FeedbackWorker(ProcessWorker):
    def __init__(
        self,
        run_files,
        pick_chs,
        stim_interval,
        stim_labels,
        srate,
        lsl_source_id,
        timeout,
        worker_name,
    ):
        self.run_files = run_files
        self.pick_chs = pick_chs
        self.stim_interval = stim_interval
        self.stim_labels = stim_labels
        self.srate = srate
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        # train model
        print("Train model process complete.")

    def consume(self, data):
        # online process
        data = np.array(data, dtype=np.float64).T
        print(data.shape)
        data = data[self.pick_chs]
        print(data.shape)

    def post(self):
        pass


if __name__ == "__main__":
    srate = 1000                # Sample rate EEG amplifier
    stim_interval = [0.0, 0.5]
    stim_labels = [1]           # Label types

    # Data path
    cnts = 1
    filepath = "data\\train\\sub1"
    runs = list(range(1, cnts + 1))
    run_files = ["{:s}\\{:d}.cnt".format(filepath, run) for run in runs]

    # pick_chs can also use the channel indexs directly
    pick_chs = ["CPz", "PZ", "O1", "OZ", "O2"]

    # Set HTOnlineSystem parameters
    ht = HTOnlineSystem(
        device_address=("192.168.1.110", 7110),
        srate=srate,
        packet_samples=100,
        num_chans=32,
    )

    # Start tcp connection with ht
    ht.connect_tcp()  
 
    # If pick_chs is a list of channel names, the tcp connection must be established first and then the worker is initialized.
    # Else if pick_chs is a list of channel indexs, the following lookup index code is not needed, and initializing the worker 
    # can be done before the tcp connection
    all_name_chs = ht.get_name_chans()
    index = []
    for ch in pick_chs:
        try:
            index.append(all_name_chs.index(ch))
        except ValueError:
            print("Channel not found in the setting.")

    lsl_source_id = "meta_online_worker"
    feedback_worker_name = "feedback_worker"

    worker = FeedbackWorker(
        run_files=run_files,
        pick_chs=index,
        stim_interval=stim_interval,
        stim_labels=stim_labels,
        srate=srate,
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name,
    )
    marker = Marker(interval=stim_interval, srate=srate, events=[1])

    # Start acquire data from ht
    ht.start_acq()  

    # Start online data processing
    ht.register_worker(feedback_worker_name, worker, marker)
    ht.up_worker(feedback_worker_name)  

    time.sleep(0.5)

    # Press any key to terminate an online process
    input("press any key to close\n")
    ht.down_worker(feedback_worker_name)
    
    time.sleep(1)

    # Stop online data retriving of ht
    ht.stop_acq()
    ht.close_connection()
    ht.clear()
    print("bye")
