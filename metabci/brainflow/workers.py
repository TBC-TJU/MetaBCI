# -*- coding: utf-8 -*-
# License: MIT License
"""
Many Workers.

"""
from typing import Optional, Any
from abc import abstractmethod
import os, multiprocessing, queue
from .logger import get_logger

logger = get_logger('worker')

class ProcessWorker(multiprocessing.Process):
    """Online processing.
    -author: Lichao Xu
    -Created on: 2021-04-01
    -update log: 
        2022-08-10 by Wei Zhao
    """
    def __init__(self, timeout: float = 1e-3, name: Optional[str] = None):
        multiprocessing.Process.__init__(self)
        self.daemon = False
        self._exit = multiprocessing.Event()
        self._in_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self.timeout = timeout
        self.worker_name = name

    def put(self, data):
        logger.info('put samples in worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
        self._in_queue.put(data)

    def run(self):
        logger.info('start worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
        self._exit.clear()
        logger.info('pre hook executed in worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
        self.pre()
        self.clear_queue()
        while not self._exit.is_set():
            try: 
                data = self._in_queue.get(timeout=self.timeout)
                logger.info('consume samples in worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
                self.consume(data)
            except queue.Empty:
                # if queue is empty, loop to wait for next data until exiting
                pass
        logger.info('post hook executed in worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
        self.post()
        self.clear_queue()
        logger.info('worker{} exit'.format(self.worker_name if self.worker_name else os.getpid()))

    @abstractmethod
    def pre(self):
        pass

    @abstractmethod
    def consume(self, data):
        pass

    @abstractmethod
    def post(self):
        pass

    def stop(self):
        logger.info('stop worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
        self._exit.set()

    def settimeout(self, timeout=0.01):
        self.timeout = timeout

    def clear_queue(self):
        logger.info('clearing queue items in worker-{}'.format(self.worker_name if self.worker_name else os.getpid()))
        while True:
            try:
                self._in_queue.get(timeout=self.timeout)
            except queue.Empty:
                break
        logger.info('all queue items in worker-{} are cleared'.format(self.worker_name if self.worker_name else os.getpid()))
