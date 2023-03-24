# -*- coding: utf-8 -*-
"""
Amplifiers.

"""
import datetime
import socket
import struct
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pylsl
import queue

from .logger import get_logger
from .workers import ProcessWorker

logger_amp = get_logger("amplifier")
logger_marker = get_logger("marker")


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
        """Ring buffer object based on python deque data
        structure to store data.

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

    def __init__(
            self, interval: list, srate: float, events: Optional[List[int]] = None
    ):
        self.events = events
        if events is not None:
            self.interval = [int(i * srate) for i in interval]
            self.latency = 0 if self.interval[1] <= 0 else self.interval[1]
            max_tlim = max(0, self.interval[0], self.interval[1])
            min_tlim = min(0, self.interval[0], self.interval[1])
            size = max_tlim - min_tlim
            if min_tlim >= 0:
                self.epoch_ind = [self.interval[0], self.interval[1]]
            else:
                self.epoch_ind = [
                    self.interval[0] - min_tlim,
                    self.interval[1] - min_tlim,
                ]
        else:
            # continous mode
            self.interval = [int(i * srate) for i in interval]
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
                    # [str(event), str(datetime.datetime.now())])
                    # .encode()).hexdigest()
                    new_key = "".join(
                        [
                            str(event),
                            datetime.datetime.now().strftime("%Y-%m-%d \
                                -%H-%M-%S"),
                        ]
                    )
                    self.countdowns[new_key] = self.latency + 1
                    logger_marker.info("find new event {}".format(new_key))
                self.is_rising = False
            elif event == 0:
                self.is_rising = True
        else:
            if "fixed" not in self.countdowns:
                self.countdowns["fixed"] = self.latency

        drop_items = []
        # update countdowns
        for key, value in self.countdowns.items():
            value = value - 1
            if value == 0:
                drop_items.append(key)
                logger_marker.info("trigger epoch for event {}".format(key))
            self.countdowns[key] = value

        for key in drop_items:
            del self.countdowns[key]
        if drop_items and self.isfull():
            return True
        return False

    def get_epoch(self):
        """Fetch data from buffer."""
        data = super().get_all()
        return data[self.epoch_ind[0]: self.epoch_ind[1]]


class TffMarker(RingBuffer):
    __EVENTS_ID = [12, 13, 14, 15, 16, 52, 53, 54, 55, 56]

    def __init__(self, sample_rate: int = 1000):
        self.interval = [0, 6] * sample_rate
        self.sample_rate = sample_rate
        max_size = self.interval[1] - self.interval[0]
        self.cur_event = 0
        super().__init__(size=max_size)

    def __call__(self, event: int) -> bool:
        m_event = int(event)
        if m_event in self.__EVENTS_ID:
            self.cur_event = m_event
            return True
        else:
            return False

    def get_epoch(self):
        index = self.__EVENTS_ID.index(self.cur_event)
        data = super().get_all()
        if index % 5 == 0:
            return data[int(-1 * self.sample_rate * 2):]
        elif index % 5 == 1:
            return data[int(-1 * self.sample_rate * 2.4):]
        elif index % 5 == 2:
            return data[int(-1 * self.sample_rate * 2.8):]
        elif index % 5 == 3:
            return data[int(-1 * self.sample_rate * 3.2):]
        elif index % 5 == 4:
            return data[int(-1 * self.sample_rate * 3.6):]


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
        """the minimal recv data function, usually a package."""
        pass

    def start(self):
        """start the loop."""
        for work_name in self._workers:
            logger_amp.info("clear marker buffer")
            self._markers[work_name].clear()
        logger_amp.info("start the loop")
        self._t_loop = threading.Thread(target=self._inner_loop,
                                        name="main_loop")
        self._t_loop.start()

    def _inner_loop(self):
        """Inner loop in the threading."""
        self._exit.clear()
        logger_amp.info("enter the inner loop")
        while not self._exit.is_set():
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples)
            except Exception:
                pass
        logger_amp.info("exit the inner loop")

    def stop(self):
        """stop the loop."""
        logger_amp.info("stop the loop")
        self._exit.set()
        logger_amp.info("waiting the child thread exit")
        self._t_loop.join()
        self.clear()

    def _detect_event(self, samples):
        """detect event label"""
        for work_name in self._workers:
            logger_amp.info("process worker-{}".format(work_name))
            marker = self._markers[work_name]
            worker = self._workers[work_name]
            for sample in samples:
                marker.append(sample)
                if marker(sample[-1]) and worker.is_alive():
                    worker.put(marker.get_epoch())

    def up_worker(self, name):
        logger_amp.info("up worker-{}".format(name))
        self._workers[name].start()

    def down_worker(self, name):
        logger_amp.info("down worker-{}".format(name))
        self._workers[name].stop()
        self._workers[name].clear_queue()

    def register_worker(self, name: str,
                        worker: ProcessWorker,
                        marker: RingBuffer):
        logger_amp.info("register worker-{}".format(name))
        self._workers[name] = worker
        self._markers[name] = marker

    def unregister_worker(self, name: str):
        logger_amp.info("unregister worker-{}".format(name))
        del self._markers[name]
        del self._workers[name]

    def clear(self):
        logger_amp.info("clear all workers")
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
        "stop_connect": b"CTRL\x00\x01\x00\x02\x00\x00\x00\x00",
        "start_acq": b"CTRL\x00\x02\x00\x01\x00\x00\x00\x00",
        "stop_acq": b"CTRL\x00\x02\x00\x02\x00\x00\x00\x00",
        "start_trans": b"CTRL\x00\x03\x00\x03\x00\x00\x00\x00",
        "stop_trans": b"CTRL\x00\x03\x00\x04\x00\x00\x00\x00",
        "show_ver": b"CTRL\x00\x01\x00\x01\x00\x00\x00\x00",
        "show_edf": b"CTRL\x00\x03\x00\x01\x00\x00\x00\x00",
        "start_imp": b"CTRL\x00\x02\x00\x03\x00\x00\x00\x00",
        "req_version": b"CTRL\x00\x01\x00\x01\x00\x00\x00\x00",
        "correct_dc": b"CTRL\x00\x02\x00\x05\x00\x00\x00\x00",
        "change_setup": b"CTRL\x00\x02\x00\x04\x00\x00\x00\x00",
    }

    def __init__(
            self,
            device_address: Tuple[str, int] = ("127.0.0.1", 4000),
            srate: float = 1000,
            num_chans: int = 68,
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.neuro_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # the size of a package in neuroscan data is
        # srate/25*(num_chans+1)*4 bytes
        self.pkg_size = srate / 25 * (num_chans + 1) * 4
        self.timeout = 2 * 25 / self.srate

    def _unpack_header(self, b_header):
        ch_id = struct.unpack(">4s", b_header[:4])
        w_code = struct.unpack(">H", b_header[4:6])
        w_request = struct.unpack(">H", b_header[6:8])
        pkg_size = struct.unpack(">I", b_header[8:])
        return (ch_id[0].decode("utf-8"), w_code[0], w_request[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        fmt = ">" + str((num_chans + 1) * 4) + "B"
        samples = (
            np.array(list(struct.iter_unpack(fmt, b_data)), dtype=np.uint8)
            .view(np.int32)
            .astype(np.float64)
        )
        samples[:, -1] = samples[:, -1] - 65280
        samples[:, :-1] = samples[:, :-1] * 0.0298 * 1e-6
        return samples.tolist()

    def _recv(self, num_bytes):
        fragments = []
        b_count = 0
        while b_count < num_bytes:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b"".join(fragments)
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
        if method == "connect":
            self.neuro_link.connect(self.device_address)
        elif method == "start_acquire":
            self.send(self._COMMANDS["start_acq"])
            self.set_timeout(None)
            self.recv()
            self.recv()
            self.set_timeout(self.timeout)
        elif method == "stop_acquire":
            self.set_timeout(None)
            self.send(self._COMMANDS["stop_acq"])
            self.recv()
            self.recv()
            self.set_timeout(self.timeout)
        elif method == "start_transport":
            self.send(self._COMMANDS["start_trans"])
            time.sleep(1e-2)
            self.start()
        elif method == "stop_transport":
            self.send(self._COMMANDS["stop_trans"])
            self.stop()
        elif method == "disconnect":
            self.send(self._COMMANDS["stop_connect"])
            if self.neuro_link:
                self.neuro_link.close()
                self.neuro_link = None

    def connect_tcp(self):
        self.neuro_link.connect(self.device_address)

    def start_acq(self):
        self.send(self._COMMANDS["start_acq"])
        self.set_timeout(None)
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def stop_acq(self):
        self.set_timeout(None)
        self.send(self._COMMANDS["stop_acq"])
        self.recv()
        self.recv()
        self.set_timeout(self.timeout)

    def start_trans(self):
        self.send(self._COMMANDS["start_trans"])
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.send(self._COMMANDS["stop_trans"])
        self.stop()

    def close_connection(self):
        self.send(self._COMMANDS["stop_connect"])
        if self.neuro_link:
            self.neuro_link.close()
            self.neuro_link = None


class Neuracle(BaseAmplifier):
    """ An amplifier implementation for neuracle devices.
    -author: Jie Mei
    -Created on: 2022-12-04

    Brief introduction:
    This class is a class for get package data from Neuracle device. To use
    this class, you must start the Neusen W software first, and then click
    the DataService icon on the right part and set parameter. The default
    port is 8712, and you do not need to modifiy it.
    (warning, this class was developed under Newsen W 2.0.1 version, we are
    not sure if it supports the newer version. You could ask for support
    from the Neuracle company.)

    Args:
        device_address: (ip, port)
        srate: sample rate of device, the default value of Neuracle is 1000
        num_chans: channel of data, for Neuracle, including data
                    channel and trigger channel
    """

    def __init__(self,
                 device_address: Tuple[str, int] = ('127.0.0.1', 8712),
                 srate=1000,
                 num_chans=9):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._update_time = 0.04
        self.pkg_size = int(self._update_time * 4 * self.num_chans * self.srate)

    def set_timeout(self, timeout):
        if self.tcp_link:
            self.tcp_link.settimeout(timeout)

    def recv(self):
        # wait for the socket available
        data = None
        # rs, _, _ = select.select([self.tcp_link], [], [], 9)
        try:
            raw_data = self.tcp_link.recv(self.pkg_size)
        except Exception:
            self.tcp_link.close()
            print("Can not receive data from socket")
        else:
            data, evt = self._unpack_data(raw_data)
            data = data.reshape(len(data) // self.num_chans, self.num_chans)
        return data.tolist()

    def _unpack_data(self, raw):
        len_raw = len(raw)
        event, hex_data = [], []
        # unpack hex_data in row
        hex_data = raw[:len_raw - np.mod(len_raw, 4 * self.num_chans)]
        n_item = int(len(hex_data) / 4 / self.num_chans)
        format_str = '<' + (str(self.num_chans) + 'f') * n_item
        unpack_data = struct.unpack(format_str, hex_data)

        return np.asarray(unpack_data), event

    def connect_tcp(self):
        self.tcp_link.connect(self.device_address)

    def start_trans(self):
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.stop()

    def close_connection(self):
        if self.tcp_link:
            self.tcp_link.close()
            self.tcp_link = None


class LSLInlet:
    """Base class for a intlet"""

    def __init__(self, info: pylsl.StreamInfo) -> None:
        self.inlet = pylsl.StreamInlet(
            info, max_buflen=3,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)

        self.name = info.name()
        self.channel_count = info.channel_count()

    def stream_action(self):
        pass


class DataInlet(LSLInlet):
    dtypes = [[], np.float32, np.float64, None,
              np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo) -> None:
        super().__init__(info)
        # Define two queue for storage the data retrieved from device
        # and their timestamp range.
        self.data_queue: queue.Queue[Any] = queue.Queue(3)

    def stream_action(self):
        samples, ts = self.inlet.pull_chunk(
            timeout=0.0, max_samples=40)
        if ts:
            samples = np.asarray(samples)
            ts = np.asarray(ts)
            pack_data = np.hstack((samples, ts.reshape((-1, 1))))
            self.data_queue.put(pack_data)

    def get_data(self):
        if self.data_queue.full():
            data = self.data_queue.get()
            return data
        else:
            return np.asarray([0])


class MarkerInlet(LSLInlet):
    def __init__(self, info: pylsl.StreamInfo) -> None:
        super().__init__(info)

    def stream_action(self):
        marker_value, marker_ts = self.inlet.pull_sample(0.0)
        if marker_ts:
            # cache = []
            # for content, ts in zip(marker_value, marker_ts):
            try:
                int_label = int(marker_value[0])
            except Exception:
                raise ValueError(
                    "The marker value: {} can not be \
                        typed into int".format(marker_value))
                # cache.append([int_label, ts])
            return [int_label, marker_ts]
        else:
            return []


class LSLapps():
    """An amplifier implementation for Lab streaming layer (LSL) apps.
    LSL ref as: https://github.com/sccn/labstreaminglayer
    The LSL provides many builded apps for communiacting with varities
    of devices, and some of the are EEG acquiring device, like EGI, g.tec,
    DSI and so on. For metabci, here we just provide a pathway for reading
    lsl data streams, which means as long as the the LSL providing the app,
    the metabci could support its online application. Considering the
    differences among different devices for transfering the event trigger.
    YOU MUST BE VERY CAREFUL to determine wethher the data stream reading
    from the LSL apps contains a event channel. For example, the neuroscan
    synamp II will append a extra event channel to the raw data channel.
    Because we do not have chance to test each device that LSL supported, so
    please modify this class before using with your own condition.
    """

    def __init__(self, ):
        super().__init__()
        self.marker_inlet = None
        self.data_inlet = None
        self.device_data = None
        self.marker_data = None
        self.marker_cache = list()
        self.marker_count = 0
        self.streams_count = 0
        self.pending_stream = []
        self.data_response = np.zeros(1)
        self.bg_stream_checker = pylsl.ContinuousResolver()
        time.sleep(1.5)
        self.stream_checker_threading = threading.Thread(
            target=self.stream_checker, name="stream_checker")
        self.stream_checker_threading.start()

    def stream_checker(self):
        while True:
            streams = self.bg_stream_checker.results()
            if len(streams) != self.streams_count:
                self.streams_count = len(streams)
                for info in streams:
                    if info.type() == 'Markers':
                        if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                                or info.channel_format() != pylsl.cf_string:
                            print('Invalid marker stream ' + info.name())
                        print('Adding marker inlet: ' + info.name())
                        self.marker_inlet = MarkerInlet(info)
                    elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
                            and info.channel_format() != pylsl.cf_string:
                        print('Adding data inlet: ' + info.name())
                        self.data_inlet = DataInlet(info)
                    else:
                        print('Don\'t know what to do \
                                with stream ' + info.name())
            time.sleep(0.5)

    def recv(self):
        if self.marker_inlet is not None:
            self.marker_data = self.marker_inlet.stream_action()
        # Check if there are markers retriving from the stream.
        if self.marker_data:
            self.marker_cache.append(self.marker_data)
            # print("Catch a trigger, content is: {}".format(self.marker_data))
        if self.data_inlet is not None:
            self.data_inlet.stream_action()
            self.data_response = self.data_inlet.get_data()
        # Check if there are devices data from the stream. Because we kept
        # a buffer, so the data will be delay about 80points. In case we
        # miss the labels
        if self.data_response.any():
            device_data = self.data_response
            epoch_length = device_data.shape[0]
            # Create a zero vector as label line
            label_line = np.zeros(epoch_length)
            # Find the label position
            for label in self.marker_cache:
                position = device_data[:, -1].searchsorted(label[-1])
                # The smaller index in the marker cache means a earlier label
                if position >= epoch_length:
                    # IF larger than the epoch max index, we say the timestamp
                    # is out of the range of current device epoch.
                    break
                else:
                    label_line[position] = int(label[0])
                    # print("The trigger position has been \
                    #       assigned to {}".format(position))
                    # print("LSL clock delta: \
                    #         {}".format(label[-1]-device_data[position, -1]))
                    # POP out the current index
                    self.marker_cache.remove(label)
            # Replaced the last column of device_data as the trigger column
            device_data[:, -1] = label_line
            return device_data.tolist()
        else:
            return []

    def start_trans(self):
        time.sleep(1e-2)
        self.start()

    def stop_trans(self):
        self.stop()
