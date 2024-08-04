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
import scipy.io
import os
import time
import signal


from pylsl.pylsl import StreamInlet, resolve_byprop

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
        self, interval: list, srate: float, events: Optional[List[int]] = None,
            save_data: Optional[bool] = False, info: dict = {}, clear_after_use = False,
            location=None, experiment_name: str = 'NoName', subject: int = 1
    ):
        self.events = events
        self.info = info
        self.info['events'] = self.events
        self.save_data = save_data
        self.raw_data = {}
        self.raw_data['experiment_name'] = experiment_name
        self.raw_data['subject'] = subject
        self.clear_after_use = clear_after_use
        self.location = location
        self.experiment_name = experiment_name
        self.subject = subject

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
        self.info['epoch_ind'] = self.epoch_ind
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
                            datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S"), ##优化key的名称
                        ]
                    )
                    self.countdowns[new_key] = self.latency + 1
                    logger_marker.info("find new event {}".format(new_key))
                    print("find new key")
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
            if self.save_data:
                self.raw_data[str(key)] = super().get_all()
                print(type(self.raw_data[str(key)]))
                print("data buffed")

        if drop_items and self.isfull():
            return True
        return False

    def get_epoch(self):
        """Fetch data from buffer."""
        data = super().get_all()
        if self.clear_after_use:
            self.clear() ##舍弃原来的数据, 不确定是否有影响
        return data[self.epoch_ind[0]: self.epoch_ind[1]]

    def save_as_mat(self):

        if self.location == None:
            user_home = os.path.expanduser('~')
            user_dir = os.path.join(user_home, 'AssistBCI\\Experiment_Raw_data')
            info_dir = os.path.join(user_home, 'AssistBCI\\Experiment_Raw_data_info')
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            if not os.path.exists(info_dir):
                os.makedirs(info_dir)
        else:
            user_dir = self.location + 'AssistBCI\\Experiment_Raw_data'
            info_dir = self.location + 'AssistBCI\\Experiment_Raw_data_info'

        name_mat = "{:s}\\E_{:s}_S_{:d}_R_{:s}.mat".format(user_dir, self.experiment_name, self.subject, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        name_txt = "{:s}\\E_{:s}.txt".format(info_dir, self.experiment_name)

        scipy.io.savemat(name_mat, self.raw_data)
        if os.path.exists(info_dir + '\\' +name_txt):
            try:
                with open(info_dir + '\\' +name_txt, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        elements = line.split('\n')[0]
                        elements = elements.split('=')
                        if elements[0] == 'subject':
                            subject = elements[1]
                            break

            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
            self.info['subject'] = eval(subject).append(self.info['subject'])

        else:
            self.info['subject'] = [self.info['subject']]

        filename = open(name_txt, 'w')
        for k, v in self.info.items():
            filename.write(k + ':' + str(v))
            filename.write('\n')
        filename.close()
        del self.info, self.raw_data
        print("Experiment Data Saved")

class BaseAmplifier:
    """Base Ampifier class.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log:
        2022-08-10 by Wei Zhao
    """

    def __init__(self, use_trigger=True):
        self._markers = {}
        self._workers = {}
        self._exit = threading.Event()
        self.use_trigger = use_trigger  #加入应用时无trigger分发数据

    @abstractmethod
    def recv(self):
        """the minimal recv data function, usually a package."""
        #返回list, len(list)为channel+event(trigger)
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
                if self.use_trigger:
                    if marker(sample[-1]) and worker.is_alive():
                        worker.put(marker.get_epoch())
                else:
                    if marker.isfull() and worker.is_alive():
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
                        marker: Marker):
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
            if self._markers[name].save_data:
                self._markers[name].save_as_mat()
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


class Curry8(BaseAmplifier):
    """An amplifier implementation for Curry8.
    Intercept online data.
    -author: Ziyu Zhou
    -Created on: 2023-07-07
    """

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
        ch_id = b_header[:4].decode()
        w_code = struct.unpack(">H", b_header[4:6])
        w_request = struct.unpack(">H", b_header[6:8])
        startSample = struct.unpack(">I", b_header[8:12])
        pkg_size = struct.unpack(">I", b_header[12:16])
        return (ch_id, w_code[0], w_request[0], startSample[0], pkg_size[0])

    def _unpack_data(self, num_chans, b_data):
        samples = np.frombuffer(b_data, dtype=np.float32).reshape(-1, num_chans).astype(np.float64)
        samples[:, -1] = samples[:, -1] - 65280
        return samples

    def _recv(self, num_bytes):
        fragments = b""
        b_count = 0
        while b_count < num_bytes:
            try:
                chunk = self.neuro_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments += chunk

        b_data = fragments
        return b_data

    def recv(self):
        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        if header[-1] != 0:
            b_data = self._recv(header[-1])
            if header[0] == "DATA":
                if header[1] == self.dataType("Data_Eeg") and header[2] == self.blockType("DataTypeFloat32bit"):
                    samples = self._unpack_data(self.num_chans, b_data)
                    return samples.tolist()
        return []

    def send(self, message):
        self.neuro_link.sendall(message)

    def set_timeout(self, timeout):
        if self.neuro_link:
            self.neuro_link.settimeout(timeout)

    def connect_tcp(self):
        self.neuro_link.connect(self.device_address)

    def start_acq(self):
        self.send(self.command_code("RequestAmpConnect"))
        self.set_timeout(None)

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("start_acq", header)

        self.set_timeout(self.timeout)

    def stop_acq(self):
        self.set_timeout(None)
        self.send(self.command_code("RequestAmpDisonnect"))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("stop_acq", header)

        self.set_timeout(self.timeout)

    def start_trans(self):  # send data
        self.send(self.command_code("RequestStreamingStart"))
        time.sleep(1e-2)

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("start_trans", header)

        self.start()

    def stop_trans(self):
        self.send(self.command_code("RequestStreamingStop"))
        self.stop()

    def close_connection(self):
        if self.neuro_link:
            self.neuro_link.close()
            self.neuro_link = None

    def update_basic_info(self):
        status, basicInfo, header = self.getBasicInfo()
        if status:
            self.srate = basicInfo["srate"]
            self.num_chans = basicInfo["num_chans"]
            self.basicInfo = basicInfo
            return True
        else:
            return False

    def update_channel_info(self):
        status, channelInfo, header = self.getChannelInfoList()
        if status:
            self.chanelNameList = [x["chanLabel"] for x in channelInfo]
            self.channelInfo = channelInfo
            return True
        else:
            return False

    def getBasicInfo(self):
        maxChans = 300

        # sendCommand
        self.send(self.command_code('RequestBasicInfoAcq'))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)

        if header[0] != 'DATA' \
                or header[1] != self.dataType("Data_Info") \
                or header[2] != self.infoType("InfoType_BasicInfo"):
            return 0, None, header

        # read basicInfo
        b_data = self._recv(header[-1])
        basicInfo = {
            'size': struct.unpack('<I', b_data[0:4])[0],
            'num_chans': struct.unpack('<I', b_data[4:8])[0],
            'srate': struct.unpack('<I', b_data[8:12])[0],
            'dataSize': struct.unpack('<I', b_data[12:16])[0],
            'allowClientToControlAmp': struct.unpack('<I', b_data[16:20])[0],
            'allowClientToControlRec': struct.unpack('<I', b_data[20:24])[0]
        }

        if basicInfo['num_chans'] > 0 and basicInfo['num_chans'] < maxChans and basicInfo['srate'] > 0 and (
                basicInfo['dataSize'] == 2 or basicInfo['dataSize'] == 4):
            status = 1
        else:
            status = 0

        return status, basicInfo, header

    def getChannelInfoList(self):
        numChannels = self.num_chans

        self.send(self.command_code("RequestChannelInfo"))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)

        if header[0] != 'DATA' \
                or header[1] != self.dataType("Data_Info") \
                or header[2] != self.infoType("InfoType_ChannelInfo"):
            status = 0
            infoList = None
            return status, infoList, header
        infoListRaw = self._recv(header[-1])

        offset_channelId = 0
        offset_chanLabel = offset_channelId + 4
        offset_chanType = offset_chanLabel + 80
        offset_deviceType = offset_chanType + 4
        offset_eegGroup = offset_deviceType + 4
        offset_posX = offset_eegGroup + 4
        offset_posY = offset_posX + 8
        offset_posZ = offset_posY + 8
        offset_posStatus = offset_posZ + 8
        offset_bipolarRef = offset_posStatus + 4
        offset_addScale = offset_bipolarRef + 4
        offset_isDropDown = offset_addScale + 4
        offset_isNoFilter = offset_isDropDown + 4

        chanInfoLen = offset_isNoFilter + 4
        chanInfoLen = round(chanInfoLen / 8) * 8

        infoList = []

        for i in range(numChannels):
            j = chanInfoLen * i
            chanInfo = {
                'id': struct.unpack('<I', infoListRaw[j + offset_channelId: j + offset_chanLabel])[0],
                'chanLabel': infoListRaw[j + offset_chanLabel: j + offset_chanType].replace(b'\x00', b'').decode(
                    'utf-8'),
                'chanType': struct.unpack('<I', infoListRaw[j + offset_chanType: j + offset_deviceType])[0],
                'deviceType': struct.unpack('<I', infoListRaw[j + offset_deviceType: j + offset_eegGroup])[0],
                'eegGroup': struct.unpack('<I', infoListRaw[j + offset_eegGroup: j + offset_posX])[0],
                'posX': struct.unpack('<d', infoListRaw[j + offset_posX: j + offset_posY])[0],
                'posY': struct.unpack('<d', infoListRaw[j + offset_posY: j + offset_posZ])[0],
                'posZ': struct.unpack('<d', infoListRaw[j + offset_posZ: j + offset_posStatus])[0],
                'posStatus': struct.unpack('<I', infoListRaw[j + offset_posStatus: j + offset_bipolarRef])[0],
                'bipolarRef': struct.unpack('<I', infoListRaw[j + offset_bipolarRef: j + offset_addScale])[0],
                'addScale': struct.unpack('<f', infoListRaw[j + offset_addScale: j + offset_isDropDown])[0],
                'isDropDown': struct.unpack('<I', infoListRaw[j + offset_isDropDown: j + offset_isNoFilter])[0],
                'isNoFilter': struct.unpack('<II', infoListRaw[j + offset_isNoFilter: j + chanInfoLen])
            }
            infoList.append(chanInfo)
        status = 1

        return status, infoList, header

    def get_server_version(self):
        self.send(self.command_code('RequestVersion'))

        b_header = self._recv(20)
        header = self._unpack_header(b_header)
        print("get_server_version", header)

        b_data = self._recv(header[-1])
        version = struct.unpack("<I", b_data)[0]
        return version

    def controlCode(self, type):
        if type == 'CTRL_FromServer':
            return 1
        elif type == 'CTRL_FromClient':
            return 2
        else:
            return -1

    def receiveType(self, code):
        if code == 1:
            return "StartAmplifier"
        elif code == 2:
            return "StopAmplifier"

    def requestType(self, type):
        if type == 'RequestVersion':
            return 1
        elif type == 'RequestChannelInfo':
            return 3
        elif type == 'RequestBasicInfoAcq':
            return 6
        elif type == 'RequestStreamingStart':
            return 8
        elif type == 'RequestStreamingStop':
            return 9
        elif type == 'RequestAmpConnect':
            return 10
        elif type == 'RequestAmpDisconnect':
            return 11
        elif type == 'RequestDelay':
            return 16
        else:
            return -1

    def dataType(self, type):
        if type == 'Data_Info':
            return 1
        elif type == 'Data_Eeg':
            return 2
        elif type == 'Data_Events':
            return 3
        elif type == 'Data_Impedance':
            return 4
        else:
            return -1

    def infoType(self, type):
        if type == 'InfoType_Version':
            return 1
        elif type == 'InfoType_BasicInfo':
            return 2
        elif type == 'InfoType_ChannelInfo':
            return 4
        elif type == 'InfoType_StatusAmp':
            return 7
        elif type == 'InfoType_Time':
            return 9
        else:
            return -1

    def blockType(self, t):
        d = -1
        if t == 'DataTypeFloat32bit':
            d = 1
        elif t == 'DataTypeFloat32bitZIP':
            d = 2
        elif t == 'DataTypeEventList':
            d = 3
        return d

    def command_code(self, method):
        c_chID = b"CTRL"
        w_Code = struct.pack('>H', self.controlCode('CTRL_FromClient'))
        w_Request = struct.pack('>H', self.requestType(method))
        un_Sample = struct.pack('>I', 0)
        un_Size = struct.pack('>I', 0)
        un_SizeUn = struct.pack('>I', 0)

        header = c_chID + w_Code + w_Request + un_Sample + un_Size + un_SizeUn
        return header

    def __del__(self):
        print("The session has been disconnected!")
        self.close_connection()


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
        self.pkg_size = int(self._update_time*4*self.num_chans*self.srate)

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
            data = data.reshape(len(data)//self.num_chans, self.num_chans)
        return data.tolist()

    def _unpack_data(self, raw):
        len_raw = len(raw)
        event, hex_data = [], []
        # unpack hex_data in row
        hex_data = raw[:len_raw - np.mod(len_raw, 4*self.num_chans)]
        n_item = int(len(hex_data)/4/self.num_chans)
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



class BlueBCI(BaseAmplifier):
    """An amplifier implementation for BlueBCI device.
    Intercept online data.
    -author: Qihao Xu
    -Created on: 2024-07-04
    """

    def __init__(
            self,
            device_address: Tuple[str, int] = ("127.0.0.1", 12345),
            srate: float = 1000,
            num_chans: int = 8,
            lsl_source_id: str = "trigger",
            use_trigger: bool = True
    ):
        super().__init__(use_trigger=use_trigger)
        self.device_address = device_address
        self.srate = srate
        self.num_chans = num_chans
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # the size of a package in neuroscan data is
        # 15*33bytes= 495 bytes
        self.timeout = 2 * 25 / self.srate
        self.n = 15  # n 应该根据你的实际需求来设置
        self.buffer_size = 33 * self.n
        self.tcp_link.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)

        self.lsl_source_id = lsl_source_id
        self.streams = []

        #self.rise = True


    def set_timeout(self, timeout):
        if self.tcp_link:
            self.tcp_link.settimeout(timeout)

    def recv(self):
        # wait for the socket available
        ###
        if self.use_trigger:
            if not self.streams:
                self.streams = resolve_byprop(
                    "source_id", self.lsl_source_id, timeout=5
                )  # Resolve all streams by source_id
                if self.streams:
                    self.inlet = StreamInlet(self.streams[0])
                    print("Connected to port")
                else:
                    print("Waiting for port")

        ###

        data = None
        try:
            received_data = self.tcp_link.recv(self.buffer_size)
            data_recv = np.frombuffer(received_data, dtype=np.uint8)
        except Exception:
            self.tcp_link.close()
            print("Can not receive data from socket")
        else:
            data = self._unpack_data(data_recv)
            #data = data.T
        return data.tolist()


    def _unpack_data(self, data_recv):
        bytes_to_read = len(data_recv)
        column_num = int(bytes_to_read / 33)
        data_recv1 = data_recv.reshape((33, column_num), order='F')
        data_recv2 = np.array(data_recv1).astype(np.float64)

        road = data_recv1[2:27:3, :]  # 从索引3开始每隔3取一行
        road = road.T
        # 修改data_recv2中的特定行
        data_recv2[2:27:3, :] *= 2 ** 16
        data_recv2[3:28:3, :] *= 2 ** 8
        data_recv2[4:29:3, :] *= 2 ** 0

        # 创建数据通道数组
        result_matrix = []
        for i in range(2, len(data_recv2) - 4, 3):  # 从索引2开始，步长为3，确保i+2不超出索引范围
            # 累加第i行、第i+1行和第i+2行
            sum_row = data_recv2[i] + data_recv2[i + 1] + data_recv2[i + 2]

            result_matrix.append(sum_row)

        result_matrix = np.array(result_matrix)
        result_matrix = result_matrix.T
        # 查找小于等于2^7的索引
        idx_chn = np.where(road >= 2 ** 7)

        # 小于等于2^7减去2^24
        result_matrix[idx_chn] -= 2 ** 24
        scale_fac_uVolts_per_count = 0.022351744455307063
        result_matrix = result_matrix * scale_fac_uVolts_per_count


        if (result_matrix[:, -1] > 0).any():
            samples = None
            timestamp = None
            try:
                samples, timestamp = self.inlet.pull_sample(timeout=0.05)
                print("event received", samples[0])
                if samples != None and timestamp != None:
                    for i in range(len(result_matrix[:, -1])):
                        if result_matrix[i, -1] != 0:
                            result_matrix[i, -1] = samples[0]
                            result_matrix[i + 1:, -1] = 0
                            break
                else:
                    result_matrix[:, -1] = 0

            except:
                result_matrix[:, -1] = 0

        return result_matrix

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
            del self.tcp_link


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


class HTOnlineSystem(BaseAmplifier):
    """
    An amplifier implementation for digital electroencephalograph device. It will analog amplify the collected
    EEG signals, analog filter them and convert them into digital signals. Then it is transmitted to the host
    computer EEG acquisition software through Ethernet for data display and storage.

    author: Wei Zhao <vivian@tju.edu.cn>

    Created on: 2023-12-4

    update log:


    Parameters
    ----------
    device_address : Tuple[ip : str, port : int]
        ip : IP address of the collection host computer.
        port : The port number.
    srate : float
        Sampling Rate, default is 1000.
    packet_samples: float
        The number of sampling points contained in each data packet, default is 100.
    num_chans: int
        Number of channels, default is 32.

    Attributes
    ----------
    tcp_link : socket object
        Socket object used for TCP connections.
    packet_points : int
        The number of sample points for all channels contained in the data packet.
    pkg_size : int
        The number of bytes occupied by the data packet.
    timeout: float
        Overtime time.

    Raises
    ----------
    ValueError
        Srate mismatch.
        Samples for each package mismatch.
        Num of chans mismatch.

    """

    _COMMANDS = {
        "start_acq": bytes([165, 16, 1, 90]),
        "stop_acq": bytes([165, 16, 2, 90]),
        "get_srate": bytes([165, 1, 1, 90]),
        "get_samples": bytes([165, 1, 2, 90]),
        "get_num_chs": bytes([165, 1, 3, 90]),
        "get_name_chs": bytes([165, 1, 4, 90])
    }

    def __init__(
        self,
        device_address: Tuple[str, int] = ("127.0.0.1", 4000),
        srate: float = 1000,
        packet_samples: float = 100,
        num_chans: int = 32
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = srate
        self.packet_samples = packet_samples
        self.tcp_link = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.num_chans = num_chans
        self.packet_points = (num_chans + 1) * packet_samples
        self.pkg_size = self.packet_points * 4
        self.timeout = 2 * 25 / self.srate

    def _unpack_header(self, b_header):
        """
        Unpack header.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        b_header: bytes
            Frame header to be unpacked.

        Returns
        ----------
        upack_header :  cell(header : int, attribute_id : int, attribute_num : int, pkg_size : int)
        header : int
            Frame header.
        attribute_id : int
            The attribute id.
        attribute_num : int
            Number of attribute values.
        pkg_size : int
            Number of bytes in all attribute values

        """

        header = struct.unpack("<B", b_header[:1])
        attribute_id = struct.unpack("<B", b_header[1:2])
        attribute_num = struct.unpack("<H", b_header[2:4])
        pkg_size = struct.unpack("<I", b_header[4:])
        return (header[0], attribute_id[0], attribute_num[0], pkg_size[0])

    def _unpack_data(self, b_data):
        """
        Unpack data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        b_data: bytes
            Data to be unpacked.

        Returns
        ----------
        samples :  list
            Unpacked data.

        """

        fmt = "<" + str(self.packet_points) + "f"
        samples = np.array(struct.unpack(fmt, b_data))  # 解开包
        samples = samples.reshape(-1, self.num_chans + 1)
        return samples.tolist()

    def _recv(self, num_bytes):
        """
        Receive the specified bytes of data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        num_bytes: int
            Number of bytes to accept.

        Returns
        ----------
        b_data:  bytes
            Received data of specified byte size.

        """

        fragments = []
        b_count = 0
        while b_count < num_bytes:
            try:
                chunk = self.tcp_link.recv(num_bytes - b_count)
            except socket.timeout as e:
                raise e
            b_count += len(chunk)
            fragments.append(chunk)

        b_data = b"".join(fragments)
        return b_data

    def recv(self):
        """
        The minimal recv data function, usually a package.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        samples:  bytes
            An unpacked data packet.

        """

        samples = None
        try:
            b_header = self._recv(8)
            header = self._unpack_header(b_header)
            if header[-1] == self.pkg_size:
                raw_data = self._recv(self.pkg_size)
                self._recv(1)

        except Exception:
            self.tcp_link.close()
            print("Can not receive data from socket")
        else:
            samples = self._unpack_data(raw_data)
        return samples

    def send(self, message):
        """
        Send command.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        samples: bytes
            The command bytes needed to be sent.

        Returns
        ----------

        """

        self.tcp_link.sendall(message)

    def get_srate(self):
        """
        Get the sampling rate of the device.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        srate : int
            The sampling rate of the device.

        """

        self.tcp_link.sendall(self._COMMANDS["get_srate"])
        b_data = self._recv(13)
        srate = int.from_bytes(b_data[8:10], "little")
        return srate

    def get_samples(self):
        """
        Get the number of sample points contained in each data packet of the device.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:


        Parameters
        ----------

        Returns
        ----------
        num_samples : int
            The number of sample points contained in each data packet.

        """

        self.tcp_link.sendall(self._COMMANDS["get_samples"])
        b_data = self._recv(13)
        num_samples = int.from_bytes(b_data[8:10], "little")
        return num_samples

    def get_num_chs(self):
        """
        Get the number of channels of the device.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        num_chs : int
            The number of channels of the device.

        """

        self.tcp_link.sendall(self._COMMANDS["get_num_chs"])
        b_data = self._recv(13)
        num_chs = int.from_bytes(b_data[8:10], "little")
        return num_chs

    def get_name_chans(self):
        """
        Get the channels used by the devices.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------
        chs_list : str
            The channels used by the devices.

        """

        self.tcp_link.sendall(self._COMMANDS["get_name_chs"])
        b_header = self._recv(8)
        header = self._unpack_header(b_header)
        samples = None
        attr_nums = (self.num_chans + 1) * 8
        if header[-1] == attr_nums:
            b_data = self._recv(attr_nums)
            samples = struct.unpack("<" + str(attr_nums) + "B", b_data)
            self._recv(1)  # 帧尾
        chs_list = []
        ch = ""
        for sample in samples:
            if chr(sample) == "\t":
                chs_list.append(ch)
                ch = ""
            elif chr(sample) != " ":
                ch += chr(sample)
        return chs_list

    def set_timeout(self, timeout):
        """
        Set timeout.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------
        timeout: float
            Overtime time.

        Returns
        ----------

        """

        if self.tcp_link:
            self.tcp_link.settimeout(timeout)

    def connect_tcp(self):
        """
        Establish tcp connection.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """

        self.tcp_link.connect(self.device_address)
        if self.get_srate() != self.srate:
            raise ValueError("Srate mismatch.")
        if self.get_samples() != self.packet_samples:
            raise ValueError("Samples for each package mismatch.")
        if self.get_num_chs() != self.num_chans + 1:
            raise ValueError("Num of chans mismatch.")

    def close_connection(self):
        """
        Close tcp connection.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """
        if self.tcp_link:
            self.tcp_link.close()
            self.tcp_link = None

    def start_acq(self):
        """
        Start acquiring data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """
        self.send(self._COMMANDS["start_acq"])
        time.sleep(1e-2)
        self.start()

    def stop_acq(self):
        """
        Stop acquiring data.

        author: Wei Zhao <vivian@tju.edu.cn>

        Created on: 2023-12-4

        update log:

        Parameters
        ----------

        Returns
        ----------

        """
        self.send(self._COMMANDS["stop_acq"])
        self.stop()
