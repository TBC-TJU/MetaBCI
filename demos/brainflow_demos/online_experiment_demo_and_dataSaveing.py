# -*- coding: utf-8 -*-
"""
SSAVEP Feedback on NeuroScan.

"""
import time
from pylsl import StreamInfo, StreamOutlet
from metabci.brainflow.amplifiers import NeuroScan, Marker, BlueBCI
from metabci.brainflow.workers import ProcessWorker





class FeedbackWorker(ProcessWorker):
    def __init__(self, lsl_source_id, timeout, worker_name):
        self.lsl_source_id = lsl_source_id
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        print("-------------------Entering Pre-------------------")
        info = StreamInfo(
            name='meta_feedback',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=self.lsl_source_id)
        self.outlet = StreamOutlet(info)
        print('Waiting connection brainstim...')
        while not self._exit:
            if self.outlet.wait_for_consumers(1e-3):
                # if self.outlet.have_consumers():
                # #不停寻找同lsl_source_id的刺激程序
                break
        print('Connected to brainstim')

    def consume(self, data):
        print("-------------------Entering consume-------------------")

        # if data:
        #     print("data:", data)

        print("inside consume")


        p_labels = 1
        print('return fake predict id', p_labels)

        while not self.outlet.have_consumers():
            time.sleep(0.1)
        self.outlet.push_sample([p_labels])
        print("predict label pushed")


    def post(self):
        pass


if __name__ == '__main__':
    # Sample rate EEG amplifier
    srate = 1000
    # Data epoch duration, 0.14s visual delay was taken account
    stim_interval = [0.14, 5.14]
    # Label types
    stim_labels = list(range(1, 7))

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    worker = FeedbackWorker(
        lsl_source_id=lsl_source_id,
        timeout=5e-2,
        worker_name=feedback_worker_name)

    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels, save_data=True)
    #save_data 控制marker是否保存数据， 最终保存时需调用：marker.save_as_mat()


    # worker.pre()
    # Set Neuroscan parameters
    bl = BlueBCI(
        device_address=('127.0.0.1', 12345),
        srate=srate,
        num_chans=8)

    # Start tcp connection with ns
    bl.connect_tcp()


    # Register worker for online data processing
    bl.register_worker(feedback_worker_name, worker, marker)

    # Start online data processing
    bl.up_worker(feedback_worker_name)
    time.sleep(5) #留时间做pre()

    # Start slicing data and passing data to worker
    bl.start_trans()

    input('press any key to close\n')

    marker.save_as_mat(experiment_name='test3',subject=2) #保存数据

    bl.down_worker('feedback_worker')
    time.sleep(1)

    # Stop online data retriving of ns
    bl.stop_trans()
    bl.close_connection()
    print('bye')
