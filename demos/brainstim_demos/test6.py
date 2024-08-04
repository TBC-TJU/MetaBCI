import multiprocessing
import queue
import time
from typing import Optional, Any

'''
用于测试multiprocessing.Process子类的使用方法
'''

class ProcessWorker(multiprocessing.Process):
    def __init__(self, timeout: float = 1e-5, name: Optional[str] = None):
        multiprocessing.Process.__init__(self)
        self.daemon = False
        self._exit = multiprocessing.Event()
        self._in_queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        self.timeout = timeout
        self.worker_name = name

    def run(self):
        self._exit.clear()
        while not self._exit.is_set():
            try:
                command = self._in_queue.get(timeout=self.timeout)
                print("inside:",command)
                time.sleep(0.2)
            except queue.Empty:
                pass
    def CMD(self,command):
        self._in_queue.put(command)

    def stop(self):
        self._exit.set()


if __name__ == '__main__':
    a = ProcessWorker()
    a.start()
    for i in range(10):
        print("outside: ", str(i))
        a.CMD(str(i))
        time.sleep(0.1)

    a.stop()
