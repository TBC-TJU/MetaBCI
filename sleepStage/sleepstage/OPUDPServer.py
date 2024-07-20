import csv
import os
import socket
import threading
import time

import brainflow.board_shim
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from loadcsv import loadData, stageOfNpz, predict
from metabci.brainflow.amplifiers import Marker
from MIndBridge import MindBridge, MindBridgeProcess

board_id = 532
# ip_address = "192.168.43.20"
ip_address = "192.168.41.190"
ip_port = 9530
board = None


def getEEg(select=0, begin=0):
    j = begin
    a = "x"
    b = "y"
    aa = []
    path = os.environ.get("SLEEPNET_JAR_PATH")
    data = np.loadtxt('./test.csv')  #从./test.csv读取数据
    data = data.T  #转置加载的数据
    # print(data)
    params = BrainFlowInputParams()  #设置 BrainFlow 参数并初始化一个 BoardShim 对象
    board = BoardShim(532, params)  #获取板 ID 为 532 的 EEG 通道和采样率。
    eeg_channels = board.get_eeg_channels(532)  #获得通道index
    sampling_rate = board.get_sampling_rate(532)
    # print(sampling_rate)
    # eeg_channels = [1, 2,3,4]
    o = data[eeg_channels]
    # 数据预处理 滤波p
    for count, channel in enumerate(eeg_channels):
        # plot timeseries
        DataFilter.detrend(o[channel - 1], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(o[channel - 1], sampling_rate, 3.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(o[channel - 1], sampling_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(o[channel - 1], sampling_rate, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH.value, 0)
    bb = o[2][j:j + 15000]
    aa = o[0][j:j + 15000]
    # - o[1][j:j + 15000]  # EOG

    X1 = bb
    X2 = aa
    data2 = []
    data2.append(X1)
    data2.append(X2)
    length = len(data2[select])
    if (begin > length):
        begin = length - 500
    if begin < 0:
        begin = 0
    n = begin + 500
    if n > length:
        n = length
    result = ""
    for i in np.arange(begin, n):
        if i == n - 1:
            result += ("%.3f" % data2[select][i])
        else:
            result += ("%.3f," % data2[select][i])
    return result


class ServerThreading(threading.Thread):
    def __init__(self, serverSocket: socket.socket, SrcURL, ResponseURL, function, encoding='utf-8'):
        threading.Thread.__init__(self)
        param = BrainFlowInputParams()
        param.ip_address = ip_address
        param.ip_port = ip_port

        self._serverSocket = serverSocket
        self._ScrURL = SrcURL
        self._ResponseURL = ResponseURL
        self._encoding = encoding
        self._stop_event = threading.Event()
        self.function = function
        self.board = BoardShim(board_id, param)
        self.board.set_log_level(6)
        self.flagStop = False
        self.flag = False
        self.addr = ResponseURL

        if self.function == "start":
            self.srate = 1000
            self.stim_interval = [0, 1]
            # Data path
            self.stim_labels = range(0, 255)
            self.outputPath = "test.csv"
            self.MB = MindBridge(
                device_address=(ip_address, 9530),
                srate=1000,
                num_chans=48)
            self.worker_name = 'MindBridge'
            self.worker = MindBridgeProcess(
                output_files=self.outputPath,
                srate=self.srate,
                timeout=5e-2,
                worker_name=self.worker_name)
            self.marker = Marker(interval=self.stim_interval, srate=self.srate,
                                 events=self.stim_labels)

    # def __del__(self):

    def getFunction(self):
        return self.function

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        print("开启线程....." + self.function)
        if self.function == "start":
            print("start session")
            # self.board.prepare_session()
            # #下面这几行代码是创建空的test.csv
            # # with open(r"./test.csv", mode='w', encoding='utf-8') as file:
            # #     writer = csv.writer(file)
            # #     writer.writerows([]) #清空
            #
            # print('start stream')
            # self.board.start_stream()
            # num_se = 3  # 3秒
            # sample_rate = 1000  # 采样率
            # while not self.flagStop:
            #     self.flag = False
            #     time.sleep(num_se)
            #     current_data = self.board.get_current_board_data(num_se * sample_rate)
            #     #print(self.function + "   reading...")
            #     # 以追加形式写入csv
            #     DataFilter.write_file(current_data, r'./test.csv', 'a')
            #     self.flag = True
            '''metaBCI'''
            self.MB.connect()
            # Start acquire data from ns
            self.MB.start_trans()
            #
            # Register worker for online data processing
            self.MB.register_worker(self.worker_name, self.worker, self.marker)
            # # Start online data processing
            self.MB.up_worker(self.worker_name)
            self.MB.start()

        if self.function.startswith("getEEG_"):
            res = getEEg(select=0, begin=int(self.function.split('_')[1]))
            print(np.array(res).shape)
            self._serverSocket.sendto(res.encode('utf-8'), (self._ResponseURL[0], 23456))
            self.stop()

        if self.function.startswith("getEOG_"):
            res = getEEg(select=1, begin=int(self.function.split('_')[1]))  # select = 1 读取眼电信号
            print(np.array(res).shape)
            self._serverSocket.sendto(res.encode('utf-8'), (self._ResponseURL[0], 23456))
            self.stop()

        if self.function.startswith('readStage_'):
            res = stageOfNpz(begin=int(self.function.split('_')[1]))
            print(np.array(res).shape)
            self._serverSocket.sendto(res.encode('utf-8'), (self._ResponseURL[0], 23456))
            self.stop()

        # print("任务结束......")


class UDPServer():
    def __init__(self):
        self._IPThreadDict = {}
        # 创建服务器套接字
        self._serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 获得本地主机名称
        self._host = socket.gethostname()
        # self._host = "127.0.0.1"
        # 设置一个端口
        self._port = 12345
        # 将套接字与本地主机和端口绑定
        self._serversocket.bind((self._host, self._port))
        #
        self._addrList = []
        myaddr = self._serversocket.getsockname()
        print("服务器地址：%s" % str(myaddr))

    def start(self):
        # UDP
        while True:
            # recv和recvfrom的区别是recv只返回数据，recvfrom返回数据源地址
            data, addr = self._serversocket.recvfrom(1024)
            if data.decode('utf-8') == 'start':
                self._addrList.append(addr)
                thread1 = ServerThreading(self._serversocket, data.decode('utf-8'), ResponseURL=addr,
                                          function=data.decode('utf-8'))
                self._IPThreadDict[addr] = thread1
                thread1.start()
            elif data.decode('utf-8').startswith('getEEG_'):
                thread2 = ServerThreading(self._serversocket, data.decode('utf-8'), ResponseURL=addr,
                                          function=data.decode('utf-8'))
                # self._IPThreadDict[addr] = thread2
                thread2.start()
            elif data.decode('utf-8').startswith('getEOG_'):
                thread3 = ServerThreading(self._serversocket, data.decode('utf-8'), ResponseURL=addr,
                                          function=data.decode('utf-8'))
                # self._IPThreadDict[addr] = thread3
                thread3.start()
            elif data.decode('utf-8').startswith('readStage_'):
                print("read_stage start")
                thread4 = ServerThreading(self._serversocket, data.decode('utf-8'), ResponseURL=addr,
                                          function=data.decode('utf-8'))
                # self._IPThreadDict[addr] = thread4
                thread4.start()
            else:
                if data.decode('utf-8') == 'stop':
                    for i in self._addrList:
                        thread = self._IPThreadDict.pop(i)
                        if thread.function == "start":
                            # thread.flagStop = True
                            # while not thread.flag:
                            #     time.sleep(1)
                            # print(thread.function)
                            # thread.board.stop_stream()
                            # print('stop session')
                            # thread.board.release_all_sessions()
                            # thread.stop()
                            thread.MB.stop_trans()
                            thread.MB.close_connection()
                            thread.MB.clear()
                            self._addrList.clear()
                            self._IPThreadDict.clear()

            # print(data)
            # print("received message:{0} from PORT {1} on {2}".format(data.decode(), addr[1], addr[0]))
            if data.decode('utf-8') == "Bye!":
                # print("服务器断开连接，结束服务！")
                self._serversocket.close()
                self._serversocket = None
                break

    def __del__(self):
        if self._serversocket is not None:
            self._serversocket.close()


def main():
    UDPServer().start()


if __name__ == '__main__':
    main()
