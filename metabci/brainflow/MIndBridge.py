import struct
import csv

import numpy as np
from metabci.brainflow.amplifiers import BaseAmplifier,Marker
from metabci.brainflow.workers import ProcessWorker
from metabci.brainflow.logger import get_logger
from typing import List, Optional, Tuple, Dict, Any
import socket
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter

logger_amp = get_logger("amplifier")
logger_marker = get_logger("marker")

ipaddress='192.168.74.190'


class MindBridge(BaseAmplifier):
    def __init__(
            self,
            device_address: Tuple[str, int] = ("127.0.0.1", 9530),
            srate: float = 1000,
            num_chans: int = 48,
    ):
        super().__init__()
        self.device_address = device_address[0]
        self.srate = srate
        self.num_chans = num_chans
        self.MindBridge_link = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._update_time = 0.3
        self.params = BrainFlowInputParams()
        self.params.ip_port = device_address[1]
        self.params.ip_address = self.device_address
        self.board = BoardShim(532, self.params)
        self.pkg_size = int(self._update_time * self.srate)
        self.board.disable_board_logger()


    def recv(self):
        data = None
        try:
            time.sleep(self._update_time)
            data = self.board.get_current_board_data(int(self._update_time*self.srate))
            # data1 = data[1:data.shape[0]]
            # data2 = data[0]
            # data = np.vstack((data1, data2))
            #print('recv')
        except Exception:
            self.board.release_session()
            print("Can not receive data from socket")
        else:
            data=data.T
        return data.tolist()

    def _detect_event(self, samples):
        """detect event label"""
        for work_name in self._workers:
            logger_amp.info("process worker-{}".format(work_name))
            marker = self._markers[work_name]
            worker = self._workers[work_name]
            for sample in samples:
                # print(np.array(sample).shape)
                marker.append(sample)
                # flag = marker(1)
                # flag=marker(sample[-1])
                # print(flag)
                if marker(sample[0]) and worker.is_alive():
                    worker.put(marker.get_epoch())


    def connect(self):
        print("Connecting to MindBridge")
        self.board.prepare_session()

    def start_trans(self):
        time.sleep(1e-2)
        print("Start trans")
        self.board.start_stream()

    def stop_trans(self):
        self.board.stop_stream()

    def close_connection(self):
        if self.board.is_prepared():
            self.board.release_session()


class MindBridgeProcess(ProcessWorker):
    def __init__(self, output_files,srate, timeout, worker_name):
        self.output_files = output_files
        self.srate = srate
        super().__init__(timeout=timeout, name=worker_name)

    def consume(self, data):
        data=np.array(data)
        data=data.T
        # print(data.shape)
        data=np.ascontiguousarray(data)
        DataFilter.write_file(data, self.output_files, 'w')
        # with open(self.output_files, mode='w', encoding='utf-8', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data)


def main():
    # Sample rate EEG amplifier
    srate = 1000
    stim_interval = [0, 1]
    # Data path
    stim_labels=range(0,255)
    filepath = "test.csv"

    worker_name = 'MindBridge'



    worker = MindBridgeProcess(
        output_files = filepath,
        srate=srate,
        timeout=5e-2,
        worker_name=worker_name)
    marker = Marker(interval=stim_interval, srate=srate,
                    events=stim_labels)

    # worker.pre()

    # Set Neuroscan parameters
    MB = MindBridge(
        device_address=(ipaddress, 9530),
        srate=srate,
        num_chans=48)

    MB.connect()
    # Start acquire data from ns
    MB.start_trans()
    #
    # Register worker for online data processing
    MB.register_worker(worker_name, worker, marker)
    # # Start online data processing
    MB.up_worker(worker_name)
    MB.start()
    #
    # # Start slicing data and passing data to worker
    # MB.down_worker(worker_name)
    # MB.start_trans()
    #
    time.sleep(10)
    #
    #
    MB.stop()
    # MB.down_worker(worker_name)
    #
    # # Stop online data retriving of ns
    MB.stop_trans()
    MB.close_connection()
    MB.clear()
    # print('bye')



if __name__ == '__main__':
    main()


