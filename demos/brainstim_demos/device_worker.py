from abc import abstractmethod
from typing import Optional, Any
from typing import Union, Optional, Dict, List, Tuple
from multiprocessing import Process, Lock ,Event, Queue, Manager
from metabci.brainflow.amplifiers import NeuroScan, BlueBCI, Curry8, Neuracle
from metabci.brainflow.amplifiers import Marker
from functools import partial
from workers import BasicWorker, ControlWorker, EmptyWorker
import time
from copy import deepcopy


class Device(Process):

    def default(self):
        self.device_list = {
            'NeuroScan': NeuroScan,
            'BlueBCI': BlueBCI,
            'Curry8': Curry8,
            'Neuracle': Neuracle
        }
        self.worker_list = {
            "BasicWorker": BasicWorker, 'ControlWorker': ControlWorker, "EmptyWorker": EmptyWorker
        }

        self.amplifier_default_parameters = {
            'BlueBCI': self.save_hyper(device_address=('127.0.0.1', 12345), srate=1000, num_chans=8, use_trigger=False)
        }

        self.worker_default_parameters = {
            'ControlWorker': self.save_hyper(timeout=1e-5, worker_name='feedback_worker', dict=self._buffer),
            "EmptyWorker": self.save_hyper(timeout=1e-5, worker_name='training_worker'),
        }

        self.marker_default_parameters = {
            'ControlWorker': self.save_hyper(interval=[0, 5], srate=1000, save_data=False, clear_after_use=False),
            "EmptyWorker": self.save_hyper(interval=[0, 5], srate=1000, save_data=True, clear_after_use=False)
        }

    def __init__(self, dict, timeout: float = 5):
        Process.__init__(self)

        self.timeout = timeout
        self.lock = Lock()
        self._buffer = dict
        self.current_device = None
        self.workers = []

        self.default()

        self.send("device_list", [str(key) for key in self.device_list.keys()])
        self.send("worker_list", [str(key) for key in self.worker_list.keys()])



        self.amplifier_args = None
        self.amplifier_kwargs = None
        self.worker_args = None
        self.worker_kwargs = None
        self.marker_args = None
        self.marker_kwargs = None

        self.send("connect_device", None)
        self.send("device_state", 'not_connected')
        self.send("reg_worker", None)
        self.send("unreg_worker", None)
        self.send("start_worker", False)
        self.send("stop_worker", False)
        self.send('current_workers', None)

        self.send_hyper("amplifier")
        self.send_hyper("worker")
        self.send_hyper("marker")

    def save_hyper(self, *args, **kwargs):
        return args, kwargs

    def send(self, name, data):
        self.lock.acquire()
        try:
            self._buffer[name] = data
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def send_hyper(self, name, *arg, **kwargs):
        self.lock.acquire()
        try:
            self._buffer[name+'_arg'] = arg
            self._buffer[name + '_kwargs'] = kwargs
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def get(self, name):
        return self._buffer[name]

    def get_hyper(self, name):
        return self._buffer[name+'_arg'], self._buffer[name + '_kwargs']

    def run(self):
        while True:
            if self.get("connect_device") != self.current_device:
                # "connect_device" 控制当前设备， 未连接时为None，连接时为设备名称
                if self.current_device == None:
                    print("current device: None")
                    self.current_device = self.get("connect_device")
                    # self.device = partial(self.device_list[self.current_device], *self.amplifier_args, **self.amplifier_kwargs)

                    amplifier_args, amplifier_kwargs = self.get_hyper('amplifier')
                    if amplifier_args or amplifier_kwargs:
                        self.send_hyper('amplifier')
                    else:
                        try:
                            amplifier_args, amplifier_kwargs = self.amplifier_default_parameters[self.current_device]
                        except:
                            amplifier_args, amplifier_kwargs = self.save_hyper() ##返回空值

                    self.device = self.device_list[self.current_device](*amplifier_args, **amplifier_kwargs)

                    try:
                        self.device.connect_tcp()
                        self.send("device_state", 'connected')
                        print("device connected")
                    except:
                        self.current_device = None
                        self.send("device_state", 'not_connected')
                        print("Fail to Make Connection with Device")
                else:
                    print("current device:", self.current_device)
                    ##切换设备或者断开设备连接
                    # for worker in self.workers:
                    #     self.device.down_worker(worker)
                    self.device.stop_trans()
                    self.workers = []
                    self.device.close_connection()
                    self.device.clear()
                    self.current_device = None
                    del self.device
                    self.send("device_state", 'not_connected')

            elif self.get("reg_worker") != None:
                try:
                    self.device
                except:
                    break
                #先使用“reg_worker"悬挂
                try:
                    name = self.get("reg_worker")

                    worker_args, worker_kwargs = self.get_hyper('worker')
                    if worker_args or worker_kwargs:
                        print("worker hyper received")
                        self.send_hyper('worker')
                    else:
                        try:
                            worker_args, worker_kwargs = self.worker_default_parameters[name]
                        except:
                            worker_args, worker_kwargs = self.save_hyper() #返回空值

                    worker = self.worker_list[name](*worker_args, **worker_kwargs)

                    marker_args, marker_kwargs = self.get_hyper('marker')
                    if marker_args or marker_kwargs:
                        print("marker hyper received")
                        self.send_hyper('marker')
                    else:
                        try:
                            marker_args, marker_kwargs = self.marker_default_parameters[name]
                        except:
                            marker_args, marker_kwargs = self.save_hyper()
                    marker = Marker(*marker_args, **marker_kwargs)
                    self.device.register_worker(worker.worker_name, worker, marker)
                    self.device.up_worker(worker.worker_name)
                    self.workers.append(name)
                except:
                    print("Fail to reg worker")
                finally:
                    self.send("reg_worker", None)
                    self.send("current_workers", self.workers)
                    print("worker ok to start")


            elif self.get("unreg_worker") != None:
                try:
                    self.device
                except:
                    break
                name = self.get("unreg_worker")
                try:
                    self.device.down_worker(name)
                    self.device.sleep(0.2)
                    self.workers.remove(name)
                except:
                    print("Fail to unreg worker")
                finally:
                    self.send("unreg_worker", None)
                    self.send("current_workers", self.workers)


            if self.get("start_worker"):
                #再使用”start_worker"开始
                try:
                    self.device.start_trans()
                    print("worker started")
                except:
                    print("Fail to start worker")
                finally:
                    self.send("start_worker", False)


            if self.get("stop_worker"):
                try:
                    self.device
                except:
                    break
                try:
                    self.device.stop_trans()
                except:
                    print("Fail to stop worker")
                finally:
                    self.send("stop_worker", False)








