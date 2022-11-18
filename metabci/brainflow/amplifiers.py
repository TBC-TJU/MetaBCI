# -*- coding: utf-8 -*-
# License: MIT License
"""
Amplifiers.

"""
import datetime
import hashlib
import os
import socket
import struct
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
from numpy import ndarray

from .logger import get_logger
from .workers import ProcessWorker

logger_amp = get_logger('amplifier')
logger_marker = get_logger('marker')

class RingBuffer(deque):
    """Online data RingBuffer.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log: 
        None
    Parameters
    ----------
        size: int,
            Size of the RingBuffer.
    """

    def __init__(self, size=1024):
        """Ring buffer object based on python deque data structure to store data.
        
        Parameters
        ----------
        size : int, optional
            maximum buffer size, by default 1024
        """
        super(RingBuffer, self).__init__(maxlen=size)
        self.max_size = size

    def isfull(self):
        """Whether current buffer is full or not.
        
        Returns
        ----------
        boolean
        """
        return len(self) == self.max_size

    def get_all(self):
        """Access all current buffer value.
        
        Returns
        ----------
        list
            the list of current buffer
        """
        return list(self)
            
class Marker(RingBuffer):
    """Intercept online data.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log: 
        2022-08-10 by Wei Zhao
    Parameters
    ----------
        interval: list,
            Time Window.
        srate: int,
            Amplifier setting sample rate.
        events: list,
            Event label.
    """

    def __init__(self, 
        interval: list, 
        srate: float, 
        events: Optional[List[int]] = None):
        self.events = events
        if events is not None:
            self.interval = [int(i*srate) for i in interval]
            self.latency = 0 if self.interval[1] <=0 else self.interval[1]
            max_tlim = max(0, self.interval[0], self.interval[1])
            min_tlim = min(0, self.interval[0], self.interval[1])
            size = max_tlim - min_tlim
            if min_tlim >= 0:
                self.epoch_ind = [self.interval[0], self.interval[1]]
            else:
                self.epoch_ind = [self.interval[0]-min_tlim, self.interval[1]-min_tlim]
        else:
            # continous mode
            self.interval = [int(i*srate) for i in interval]
            self.latency = self.interval[1] - self.interval[0]
            size = self.latency
            self.epoch_ind = [0, size]

        self.countdowns: Dict[str, int] = {}
        self.is_rising = True
        super().__init__(size=size)

    def __call__(self, event: int):
        """Record label position.
        Parameters
        ----------
            event: int,
                Online real-time data tagging.
        """
        # add new countdown items
        if self.events is not None:
            event = int(event)
            if event != 0 and self.is_rising:
                if event in self.events:
                    # new_key = hashlib.md5(''.join(
                    #     [str(event), str(datetime.datetime.now())]).encode()).hexdigest()
                    new_key = ''.join([str(event), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')])
                    self.countdowns[new_key] = self.latency + 1
                    logger_marker.info('find new event {}'.format(new_key))
                self.is_rising = False
            elif event == 0:
                self.is_rising = True
        else:
            if 'fixed' not in self.countdowns:
                self.countdowns['fixed'] = self.latency
        
        drop_items = []
        # update countdowns
        for key, value in self.countdowns.items():
            value = value - 1
            if value == 0:
                drop_items.append(key)
                logger_marker.info('trigger epoch for event {}'.format(key))
            self.countdowns[key] = value

        for key in drop_items:
            del self.countdowns[key]
        if drop_items and self.isfull():
            return True
        return False

    def get_epoch(self):
        """Fetch data from buffer.
        """
        data = super().get_all()
        return data[self.epoch_ind[0]:self.epoch_ind[1]]
            
class BaseAmplifier:
    """Base Ampifier class. 
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log: 
        2022-08-10 by Wei Zhao
    """
    
    def __init__(self):
        self._markers = {}
        self._workers = {}
        self._exit = threading.Event()

    @abstractmethod
    def recv(self):
        """the minimal recv data function, usually a package.
        """
        pass

    def start(self):
        """start the loop.
        """
        for work_name in self._workers:
            logger_amp.info('clear marker bufer')
            self._markers[work_name].clear()
        logger_amp.info('start the loop')
        self._t_loop = threading.Thread(
            target=self._inner_loop, 
            name='main_loop')
        self._t_loop.start()

    def _inner_loop(self):
        """Inner loop in the threading.
        """
        self._exit.clear()
        logger_amp.info('enter the inner loop')
        while not self._exit.is_set():
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples)
            except:
                pass
        logger_amp.info('exit the inner loop')

    def stop(self):
        """stop the loop.
        """
        logger_amp.info('stop the loop')
        self._exit.set()
        logger_amp.info('waiting the child thread exit')
        self._t_loop.join()
        self.clear()

    def _detect_event(self, samples):
        """detect event label"""
        for work_name in self._workers:
            logger_amp.info('process worker-{}'.format(work_name))
            marker = self._markers[work_name]
            worker = self._workers[work_name]
            for sample in samples:
                marker.append(sample)
                if marker(sample[-1]) and worker.is_alive():
                    worker.put(marker.get_epoch())
    
    def up_worker(self, name):
        logger_amp.info('up worker-{}'.format(name))
        self._workers[name].start()

    def down_worker(self, name):
        logger_amp.info('down worker-{}'.format(name))
        self._workers[name].stop()
        self._workers[name].clear_queue()

    def register_worker(self, 
            name: str, 
            worker: ProcessWorker, 
            marker: Marker):
        logger_amp.info('register worker-{}'.format(name))
        self._workers[name] = worker
        self._markers[name] = marker
    
    def unregister_worker(self, 
            name: str):
        logger_amp.info('unregister worker-{}'.format(name))
        del self._markers[name]
        del self._workers[name]

    def clear(self):
        logger_amp.info('clear all workers')
        worker_names = list(self._workers.keys())
        for name in worker_names:
            self._markers[name].clear()
            self.down_worker(name)
            self.unregister_worker(name)

class NeuroScan(BaseAmplifier):
    """An amplifier implementation for NeuroScan device.
        Intercept online data.
        -author: Lichao Xu
        -Created on: 2021-04-01
        -update log: 
            2022-08-10 by Wei Zhao
    """

    _COMMANDS = {
        'stop_connect': b'CTRL\x00\x01\x00\x02\x00\x00\x00\x00',
        'start_acq': b'CTRL\x00\x02\x00\x01\x00\x00\x00\x00',
        'stop_acq': b'CTRL\x00\x02\x00\x02\x00\x00\x00\x00',
        'start_trans': b'CTRL\x00\x03\x00\x03\x00\x00\x00\x00',
        'stop_trans': b'CTRL\x00\x03\x00\x04\x00\x00\x00\x00',
        'show_ver': b'CTRL\x00\x01\x00\x01\x00\x00\x00\x00',
        'show_edf': b'CTRL\x00\x03\x00\x01\x00\x00\x00\x00',
        'start_imp': b'CTRL\x00\x02\x00\x03\x00\x00\x00\x00',
        'req_version': b'CTRL\x00\x01\x00\x01\x00\x00\x00\x00',
        'correct_dc': b'CTRL\x00\x02\x00\x05\x00\x00\x00\x00',
        'change_setup': b'CTRL\x00\x02\x00\x04\x00\x00\x00\x00'
    }

    def __init__(self, 
            device_address: Tuple[str, int] = ('127.0.0.1', 4000), 
            srate: float = 1000, 
            num_chans: int =68):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.neuro_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # the size of a package in neuroscan data is srate/25*(num_chans+1)*4 bytes
        self.pkg_size = srate/25*(num_chans+1)*4
        self.timeout = 2*25/self.srate

    def _unpack_header(self, b_header):
        ch_id = struct.unpack('>4s', b_header[:4])
        w_code = struct.unpack('>H', b_header[4:6])
        w_request = struct.unpack('>H', b_header[6:8])
        pkg_size = struct.unpack('>I', b_header[8:])
        return (ch_id[0].decode('utf-8'), w_code[0], w_request[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        fmt = '>' + str((num_chans+1)*4) + 'B'
        samples = np.array(list(struct.iter_unpack(fmt, b_data)), dtype=np.uint8).view(np.int32).astype(np.float64)
        samples[:, -1] = samples[:, -1] - 65280
        samples[:, :-1] = samples[:, :-1]*0.0298*1e-6
        return samples.tolist()
   
    def _recv(self, num_bytes):
        fragments = []
        b_count = 0
        while b_count<num_bytes:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b''.join(fragments)
        return b_data

    def recv(self):
        b_header = self._recv(12)
        header = self._unpack_header(b_header)
        samples = None
        if header[-1] != 0:
            b_data = self._recv(header[-1])
            samples = self._unpack_data(self.num_chans, b_data)
        return samples

    def send(self, message):
        self.neuro_link.sendall(message)

    def set_timeout(self, timeout):
        if self.neuro_link:
            self.neuro_link.settimeout(timeout)

    def command(self, method):
        if method == 'connect':
            self.neuro_link.connect(self.device_address)
        elif method == 'start_acquire':
            self.send(self._COMMANDS['start_acq'])
            self.set_timeout(None)
            self.recv()
            self.recv()
            self.set_timeout(self.timeout)
        elif method == 'stop_acquire':
            self.set_timeout(None)
            self.send(self._COMMANDS['stop_acq'])
            self.recv()
            self.recv()
            self.set_timeout(self.timeout)
        elif method == 'start_transport':
            self.send(self._COMMANDS['start_trans'])
            time.sleep(1e-2)
            self.start()
        elif method == 'stop_transport':
            self.send(self._COMMANDS['stop_trans'])
            self.stop()
        elif method == 'disconnect':
            self.send(self._COMMANDS['stop_connect'])
            if self.neuro_link:
                self.neuro_link.close()
                self.neuro_link = None
    
    def connect_tcp(self):
        self.neuro_link.connect(self.device_address)
    
    def start_acq(self):
        self.send(self._COMMANDS['start_acq'])
        self.set_timeout(None)
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)
    
    def stop_acq(self):
        self.set_timeout(None)
        self.send(self._COMMANDS['stop_acq'])
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def start_trans(self):
        self.send(self._COMMANDS['start_trans'])
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.send(self._COMMANDS['stop_trans'])
        self.stop()

    def close_connection(self):
        self.send(self._COMMANDS['stop_connect'])
        if self.neuro_link:
            self.neuro_link.close()
            self.neuro_link = None
