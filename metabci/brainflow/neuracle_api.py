import socket
import time
from enum import Enum
from threading import Thread, Lock
from struct import unpack
import numpy as np
import tempfile
from pyedflib import highlevel
import pickle
from io import TextIOWrapper
from pathlib import Path
from typing import Optional


class BaseBuffer:
    def __init__(self, n_chan, n_points):
        self.n_chan = n_chan     # 通道数
        self.n_points = n_points  # 缓冲区总容量（点数）
        self.buffer = np.zeros((n_chan, n_points))   # 初始化缓冲区
        self.lastPtr = 0   # 最后读取指针
        self.currentPtr = 0   # 当前写入指针
        self.nUpdate = 0   # 更新点数
        self.bufferLock = Lock()  # 缓冲区锁

    # reset buffer
    # 重置缓冲区，清空数据
    def resetBuffer(self):
        self.bufferLock.acquire()   # 获取锁
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.currentPtr = 0
        self.nUpdate = 0
        self.bufferLock.release()   # 释放锁


class DoubleBuffer(BaseBuffer):     # 双缓冲区：前端缓冲区：接收新数据；后端缓冲区：暂存数据
    """
    DoubleBuffer is a buffer with two "chunks" of buffer: one acts as "front"
    buffer who receives new data and actively refresh its content, another acts
    as "back" buffer who temporarily stores a data chunk, waits to be flushed
    into disk (or a temp file) and never to be changed.

    DoubleBuffer using two buffers to balance performance and memory usage.
    """

    def __init__(self, n_chan: int, n_points: int = 300000):
        super(DoubleBuffer, self).__init__(n_chan, n_points)
        self.backBuffer = np.zeros((n_chan, n_points))    # 后端缓冲区
        self.backBufferLock = Lock()   # 后端缓冲区锁
        self.tempfile = []    # 临时文件列表
        # 后端缓冲区是否已缓存
        self.cached = True  # indicate if a back buffer has been cached
        # 后端是否有未读取数据
        self.backBufferRemain = False  # if newly updated data remain on
        # the back buffer, i.e., not read
        # by self.getUpdate()
        self.firstTime = True  # indicate the first cache  # 首次缓存标志

    def flip(self):
        # 切换前后端缓冲区
        # print('flip')
        # check if original back buffer has been cached
        assert self.cached      # assert 条件
        # assert not self.backBufferRemain)
        # flip
        self.bufferLock.acquire()
        self.backBufferLock.acquire()
        self.backBuffer = self.buffer    # 交换缓冲区
        self.cached = False
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.backBufferRemain = True
        self.currentPtr = 0
        self.bufferLock.release()
        self.backBufferLock.release()
        # caching data
        Thread(target=self.caching).start()  # 启动缓存线程

    def caching(self):
        # 将后端缓冲区数据写入临时文件
        self.backBufferLock.acquire()
        assert not self.cached
        tf = tempfile.TemporaryFile()
        '''
        if self.firstTime:
            data = self.backBuffer
        else:
            self.tempfile.seek(0)
            tem = np.load(self.tempfile, allow_pickle=True)
            data = np.hstack([tem, self.backBuffer])
        self.tempfile.seek(0)
        data.dump(self.tempfile)
        '''
        self.backBuffer.dump(tf)
        self.cached = True
        self.tempfile.append(tf)
        self.backBufferLock.release()
        self.firstTime = False

    def appendBuffer(self, data):
        # 追加数据到前端缓冲区
        """
        Append buffer and update current pointer.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            New data chunk to be updated.
        """
        # CAUTION: Cannot write anything to backbuffer!
        n = data.shape[1]   # 获取新数据的列数（时间点数）
        if self.currentPtr + n > self.n_points:  # full buffer
            # buffering data at the tail first
            tail = data[:, :self.n_points - self.currentPtr]
            data = data[:, self.n_points - self.currentPtr:]
            n -= self.n_points - self.currentPtr
            self.bufferLock.acquire()
            self.buffer[:, self.currentPtr:] = tail
            self.bufferLock.release()
            # then flip two buffer
            self.flip()
        # XXX: do not consider the situation that received data chunk is much
        #      longer than self.n_points. Please use a large enough buffer
        #      length.
        self.bufferLock.acquire()
        self.buffer[:, self.currentPtr:self.currentPtr + n] = data
        self.currentPtr = self.currentPtr + n
        self.nUpdate = self.nUpdate + n
        self.bufferLock.release()

    def getUpdate(self):
        self.bufferLock.acquire()
        if self.backBufferRemain:  # first to read backbuffer
            self.backBufferLock.acquire()
            data = self.backBuffer[:, self.lastPtr:]   # 读取后端缓冲区的剩余数据
            self.backBufferRemain = False   # 标记后端数据已读
            self.backBufferLock.release()
            data = np.hstack([data, self.buffer[:, :self.currentPtr]])
        else:
            data = self.buffer[:, self.lastPtr:self.currentPtr]   # 直接读取新增数据
        self.lastPtr = self.currentPtr
        self.nUpdate = 0  # 重置更新计数器
        self.bufferLock.release()
        return data

    def getData(self):
        # 获取双缓冲区中 所有已存储的数据
        self.bufferLock.acquire()
        self.backBufferLock.acquire()
        # 1. 从临时文件加载历史数据
        diskData = []
        for tf in self.tempfile:
            tf.seek(0)
            diskData.append(np.load(tf, allow_pickle=True))
        # 2. 加载内存数据
        if self.cached:    # 表示后端缓冲区(backBuffer)已持久化到磁盘，只需读取前端缓冲区(buffer)的数据。
            '''
            self.tempfile.seek(0)
            diskData = np.load(self.tempfile, allow_pickle=True)
            '''
            memData = self.buffer[:, :self.currentPtr]
        else:
            '''
            self.tempfile.seek(0)
            diskData = np.load(self.tempfile, allow_pickle=True)
            '''
            memData = np.hstack([
                self.backBuffer, self.buffer[:, :self.currentPtr]])
        data = np.hstack([*diskData, memData])
        self.backBufferLock.release()
        self.bufferLock.release()
        return data


class RingBuffer(BaseBuffer):
    # 环形缓冲区实现
    def appendBuffer(self, data):
        """
        Append buffer and update current pointer.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
            New data chunk to be updated.
        """
        n = data.shape[1]
        self.bufferLock.acquire()
        self.buffer[:, np.mod(np.arange(self.currentPtr, self.currentPtr + n),
                              self.n_points)] = data
        self.currentPtr = np.mod(self.currentPtr + n - 1, self.n_points) + 1
        self.nUpdate = self.nUpdate + n  # 记录新增数据点数
        self.bufferLock.release()

    def getUpdate(self):
        # 从环形缓冲区中获取 自上次调用以来新增的数据块
        self.bufferLock.acquire()
        if self.nUpdate <= self.n_points:
            if self.lastPtr <= self.currentPtr:
                data = self.buffer[:, self.lastPtr:self.currentPtr]
            else:
                data = np.hstack([self.buffer[:, self.lastPtr:], self.buffer[:, :self.currentPtr]])
        else:
            data = np.hstack([self.buffer[:, self.currentPtr:], self.buffer[:, :self.currentPtr]])
        self.lastPtr = self.currentPtr
        self.nUpdate = 0
        self.bufferLock.release()
        return data

    def getData(self):
        self.bufferLock.acquire()
        data = np.hstack([self.buffer[:, self.currentPtr:], self.buffer[:, :self.currentPtr]])
        self.bufferLock.release()
        return data


def resolveMeta(raw: bytes) -> dict:
    """
    具体的解析meta包的过程
    :param raw:meta包的原始数据，一堆byte
    :return:
    """
    # 解析HeadLength
    headerLength = int.from_bytes(raw[2:6], byteorder="little", signed=False)
    head = raw[:headerLength]
    # 解析HeadToken,HeaderLength,TotalLength,Flag,ModuleCount(这些合起来总长为HeaderLength)
    # <代表以little-endian方式解析
    # H代表1个unsigned short(2个Byte)
    # 4I代表4个unsigned int(4个Byte)
    # 具体见 https://docs.python.org/3/library/struct.html
    _, headerLength, totalLength, flag, moduleCount = unpack("<H4I", head)
    # 解析剩余部分
    body = raw[headerLength:]
    # 探头数量
    D = moduleCount
    # 每个设备数据包的起点偏移量,unpack返回的是tuple
    moduleOffsets = unpack(f"<{D}I", body[: 4 * D])
    # 每个探头的byte数据
    eachModuleData = []
    for m in range(D):
        offset = moduleOffsets[m]
        if m < D - 1:
            # 前D-1个探头的数据范围是这个探头的起点偏移量到下一个探头的起点偏移量
            end = moduleOffsets[m + 1]
        else:
            # 最后一个探头的数据范围是起点偏移量到整个meta包的末尾去除TailToken
            end = totalLength - 2
        eachModuleData.append(raw[offset:end])
    # 解析每个探头的meta数据，key是SN号，value是具体的meta数据
    modules = {}
    for m in range(D):
        module = resolveMetaEachModule(eachModuleData[m])
        key = module['serialNumber']
        modules[key] = module
    # 返回的结果
    result = {
        "flag": flag,
        "moduleCount": moduleCount,
        "modules": modules
    }
    return result


def resolveMetaEachModule(fragment: bytes) -> dict:
    """
    解析meta包中每个探头的信息
    :param fragment:每个探头的原始数据，一堆byte
    :return:
    """
    # 解析PersonName,ModuleName,ModuleType,SerialNumber,ChannelCount
    personName, moduleName, moduleType, serialNumber, channelCount = unpack("<30s30s30s2I", fragment[:98])
    # 去除首尾空格
    personName = personName.decode("utf8").strip("\x00")
    moduleName = moduleName.decode("utf8").strip("\x00")
    moduleType = moduleType.decode("utf8").strip("\x00")
    # 通道名称
    channelNames = list(unpack("10s" * channelCount, fragment[98: 98 + 10 * channelCount]))
    channelNames = [b.decode("utf8").strip("\x00") for b in channelNames]
    # 通道类型
    channelTypes = list(unpack("10s" * channelCount, fragment[98 + 10 * channelCount: 98 + 20 * channelCount]))
    channelTypes = [b.decode("utf8").strip("\x00") for b in channelTypes]
    # 采样率
    sampleRates = list(unpack(f"<{channelCount}I", fragment[98 + 20 * channelCount: 98 + 24 * channelCount]))
    # 各个通道的数据量
    dataCountPerChannel = list(unpack(f"<{channelCount}I", fragment[98 + 24 * channelCount: 98 + 28 * channelCount]))
    # 各个通道的最大数字值
    maxDigital = list(unpack(f"<{channelCount}i", fragment[98 + 28 * channelCount: 98 + 32 * channelCount]))
    # 各个通道的最小数字值
    minDigital = list(unpack(f"<{channelCount}i", fragment[98 + 32 * channelCount: 98 + 36 * channelCount]))
    # 各个通道的最大模拟值
    maxPhysical = list(unpack(f"<{channelCount}f", fragment[98 + 36 * channelCount: 98 + 40 * channelCount]))
    # 各个通道的最小模拟值
    minPhysical = list(unpack(f"<{channelCount}f", fragment[98 + 40 * channelCount: 98 + 44 * channelCount]))
    # 各个通道的增益
    gain = list(unpack(f"{channelCount}c", fragment[98 + 44 * channelCount: 98 + 45 * channelCount]))
    # 返回的结果
    result = {"personName": personName,
              "moduleName": moduleName,
              "moduleType": moduleType,
              "serialNumber": serialNumber,
              "channelCount": channelCount,
              "channelNames": channelNames,
              "channelTypes": channelTypes,
              "sampleRates": sampleRates,
              "dataCountPerChannel": dataCountPerChannel,
              "maxDigital": maxDigital,
              "minDigital": minDigital,
              "maxPhysical": maxPhysical,
              "minPhysical": minPhysical,
              "gain": gain}
    return result


def isChannelNotAllEEG(channelTypes: list) -> bool:
    """
    判断是否所有通道都是非EEG类型
    :param channelTypes:所有通道的类型
    :return:
    """
    for channelType in channelTypes:
        # 有一个通道是EEG就返回False
        if channelType == 'EEG':
            return False
    # 所有通道都不是EEG就返回True
    return True


def resolveData(raw: bytes, meta: dict) -> dict:
    """
    解析数据包的具体过程
    :param raw:数据包原始数据，一堆byte
    :param meta:解析好的meta包
    :return:
    """
    # 包头的总长度，固定为30
    headerLength = int.from_bytes(raw[2:6], byteorder="little", signed=False)
    # 解析HeadToken,HeaderLength,TotalLength,StartTimestamp,TimeStampLength,TriggerCount,Flag,ModuleCount
    head = raw[:headerLength]
    # HeadToken为unsigned short，其余为unsigned int
    _, headerLength, totalLength, startTimeStamp, timeStampLength, triggerCount, flag, moduleCount = unpack("<H7I", head)
    # 解析剩余部分
    body = raw[headerLength:]
    # 协议中有使用D和T，保持一致
    D = moduleCount
    T = triggerCount
    # 每个探头的在数据包中的偏移量，moduleOffsets是个tuple
    moduleOffsets = unpack(f"<{D}I", body[: 4 * D])
    # 前面D-1个探头的数据范围是这个探头的偏移起点到下一个探头的偏移起点
    # 最后一个探头的数据范围是偏移起点到数据末尾去除TriggerTimestamps,Triggers,TailToken三项
    eachModuleData = []
    for m in range(D):
        offset = moduleOffsets[m]
        if m < D - 1:
            end = moduleOffsets[m + 1]
        else:
            # TriggerTimestamps,Triggers,TailToken三项分别长4*T,30*T,2
            end = totalLength - 2 - 34 * T
        eachModuleData.append(raw[offset:end])
    # 每个探头的数据，包括sn号、Bitmask和Datas
    modules = {}
    for m in range(D):
        module = resolveDataEachModule(eachModuleData[m], meta)
        key = module['serialNumber']
        modules[key] = module
    # 返回的结果
    result = {
        "flag": flag,
        "startTimeStamp": startTimeStamp,
        "timeStampLength": timeStampLength,
        "moduleCount": moduleCount,
        "modules": modules
    }
    return result


def resolveDataEachModule(fragment: bytes, meta: dict) -> dict:
    """
    整体转发解析每个探头数据的过程
    :param fragment:这个探头的原始数据，一堆byte
    :param meta:解析好的meta数据
    :return:
    """
    # 这个探头的SN号
    serialNumber = int.from_bytes(fragment[:4], byteorder="little", signed=False)
    # 通道数量
    N = meta["modules"][serialNumber]["channelCount"]
    # 每个通道的数据点数
    dataCountPerChannel = meta["modules"][serialNumber]["dataCountPerChannel"]
    # 每个通道是否有值
    bitmask = list(unpack(f"{N}?", fragment[4:4 + N]))
    # data是个list，shape为:通道数*每个通道点数
    data = []
    raw = list(unpack(f"<{sum(dataCountPerChannel)}f", fragment[4 + N:]))
    # 把一段很长的连续的原始数据raw，根据每个通道的点数切成通道数*每个通道点数
    cursor = 0
    for count in dataCountPerChannel:
        data.append(raw[cursor: cursor + count])
        cursor += count
    # 返回的结果
    result = {
        "serialNumber": serialNumber,
        "bitmask": bitmask,
        "data": data
    }
    return result


class ConnectState(Enum):
    # the first state with nothing ready
    NOTCONNECT = 0
    # already connect the data server, demonstrating at least
    # there is an available TCP port
    CONNECTED = 1
    # successfully receive and resolve the META packet, and
    # open the data flow in order to stabilize it. (data will
    # not be stored into buffer)
    READY = 2
    # receiving data at present
    RUNNING = 3
    # 中止连接
    ABORT = 4


class DataServerThread:
    """
    用户就只需调用这个类
    """

    def __init__(self, sample_rate: int = 1000, t_buffer: float = 60):
        """
        初始化
        :param sample_rate:采样率
        :param t_buffer:buffer的长度(秒)
        """
        self.__binaryLog: Optional[TextIOWrapper] = None
        self.__triggerLog: Optional[TextIOWrapper] = None
        # 采样率
        self.sample_rate = sample_rate
        # buffer长度(秒)
        self.t_buffer = t_buffer
        # 初始化为不连接
        self.state = ConnectState.NOTCONNECT
        # 在正式接收数据前有一个过渡期让数据稳定
        self.stabilizeCount = 0
        self.pointsForStabilize = 50
        # 开始和结束的时间戳
        self.firstTimestamp = -1
        self.lastTimestamp = -1
        # 是否已经接收到meta包
        self.__hasMeta = False
        # socket连接使用的buffer的锁
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socketBuffer = bytes()
        self.sockBufLock = Lock()
        # 缓存单探头转发数据包
        self.single_module_data_buffer = []
        # 缓存单探头转发trigger包
        self.single_module_trigger_buffer = []
        # 缓存单探头转发数据包上限
        self.max_single_packet = 20
        # 总共收到的包总数
        self.packet_count = 0
        # 收到的包的时间戳,data和trigger分开
        self.timeStamp = {'data': [], 'trigger': []}
        # # 用于实时输出单探头转发下trigger，和jellyfish对比
        # self.trigger_count = 0

    def setBinaryLog(self, fp: Path):
        if self.__binaryLog is None:
            self.__binaryLog = open(fp, mode="w", encoding="utf8", newline="\n")

    def setTriggerLog(self, fp: Path):
        if self.__triggerLog is None:
            self.__triggerLog = open(fp, mode="w", encoding="utf8", newline="\n")
            self.__triggerLog.write("timestamp,val\n")

    def __del__(self):
        if self.__binaryLog is not None:
            self.__binaryLog.close()
        if self.__triggerLog is not None:
            self.__triggerLog.close()

    def connect(self, hostname: str = '127.0.0.1', port: int = 8712):
        """
        和JellyFish进行连接
        :param hostname:JellyFish软件运行的电脑的ip，如果在本机运行，就固定为127.0.0.1
        :param port:JellyFish开放的端口号，默认为8712
        :return:
        """
        self.hostname = hostname
        self.port = port
        # 重连次数
        reconnect_time = 0
        # 一直尝试连接下去，直到连接成功
        while self.state == ConnectState.NOTCONNECT:
            try:
                self.sock.connect((self.hostname, self.port))
                # 设置为非阻塞式接收
                self.sock.setblocking(False)
                # 设置状态为已连接
                self.state = ConnectState.CONNECTED
                # 启动读取数据的线程
                self.openStream()
                print('connect success')
            except:
                # 连接失败时输出失败次数
                reconnect_time += 1
                print(f'connection failed, retrying for {reconnect_time} times')
                time.sleep(1)
                # 超过最大重连次数就放弃重连
                if reconnect_time >= 1:
                    break
        # 返回是否连接成功
        return self.state == ConnectState.NOTCONNECT

    def isReady(self):
        """
        判断是否已经准备好读取数据(meta包有没有解析成功)
        :return:
        """
        return self.state == ConnectState.READY

    def start(self):
        """
        开始接收转发的数据
        :return:
        """
        # 还没解析好meta包就不允许开始接收数据
        if self.state == ConnectState.NOTCONNECT or self.state == ConnectState.CONNECTED:
            raise RuntimeError("Cannot start data recording before ready.")
        # meta包已经解析好了就能就开始了
        elif self.state == ConnectState.READY:
            self.state = ConnectState.RUNNING

    def stop(self):
        """
        停止接收数据
        :return:
        """
        self.state = ConnectState.ABORT

    def openStream(self):
        """
        启动读取数据的线程
        :return:
        """
        # 收数据和解析必须放在不同线程中，不然5ms单探头转发会丢包
        Thread(target=self.readDataThread, daemon=True).start()

    def readDataThread(self):
        """
        在另一个线程中执行的接收数据函数
        :return:
        """
        # 还没连接就不能接收数据
        if self.state == ConnectState.NOTCONNECT:
            raise RuntimeError("Cannot start receiving data before connect.")
        # 一直不停接收数据直到停止接收
        self.l: int = 0
        self.msg: bytearray = bytearray()
        self.isMeta: bool = False
        while self.state != ConnectState.ABORT:
            # 具体的接收数据的函数
            self.receiveData()
            # 不能sleep，sleep 1ms对于探头转发的5ms有严重影响
            # time.sleep(0.001)

    def receiveData(self):
        """
        具体的接收数据的函数
        :return:
        """
        try:
            # 1. 匹配包头令牌
            if self.l < 2:
                metaHeadToken: bytes = bytes.fromhex('5FF5')
                dataHeadToken: bytes = bytes.fromhex('5AA5')
                headToken: bytes = self.sock.recv(2)
                self.msg.extend(headToken)
                if self.__binaryLog is not None:
                    self.__binaryLog.write(headToken.hex())
                if headToken == metaHeadToken:
                    self.isMeta = True
                    self.l += 2
                elif headToken == dataHeadToken:
                    self.l += 2
                elif len(headToken) == 0:
                    return  # 流为空
                else:
                    raise ValueError(
                        f"Invalid head token \"{headToken.hex()}\"")
            # 2. 获取包头长度
            if 2 <= self.l < 6:
                bHeaderLength: bytes = self.sock.recv(6 - self.l)
                self.l += len(bHeaderLength)
                self.msg.extend(bytes(bHeaderLength))
                if self.__binaryLog is not None:
                    self.__binaryLog.write(bytes(bHeaderLength).hex())
            # 3. 获取包长度
            if 6 <= self.l < 10:
                bTotalLength: bytes = self.sock.recv(10 - self.l)
                self.l += len(bTotalLength)
                self.msg.extend(bytes(bTotalLength))
                if self.__binaryLog is not None:
                    self.__binaryLog.write(bytes(bTotalLength).hex())
            if self.l >= 10:
                totalLength: int = int.from_bytes(
                    self.msg[6:10], byteorder="little", signed=False)
            # 4. 获取整个packet
            if 10 <= self.l < totalLength:
                bBody: bytes = self.sock.recv(totalLength - self.l)
                self.l += len(bBody)
                self.msg.extend(bytes(bBody))
                if self.__binaryLog is not None:
                    self.__binaryLog.write(bytes(bBody).hex())
            # 5. 匹配包尾令牌并提交packet
            if self.l == totalLength:
                metaTailToken: bytes = bytes.fromhex('F55F')
                dataTailToken: bytes = bytes.fromhex('A55A')
                tailToken: bytes = self.msg[-2:]
                if self.isMeta and tailToken != metaTailToken:
                    raise ValueError(
                        f"Invalid tail token \"{tailToken.hex()}\"")
                elif not self.isMeta and tailToken != dataTailToken:
                    raise ValueError(
                        f"Invalid tail token \"{tailToken.hex()}\"")
                if totalLength != len(self.msg):
                    raise ValueError(
                        'The message is not as long as it assigned!')
                self.resolve(msg=self.msg, isMeta=self.isMeta)  # 提交
                # 准备下一轮流式解析
                self.l: int = 0
                self.msg: bytearray = bytearray()
                self.isMeta: bool = False
        except BlockingIOError:
            pass

    def isSingleModule(self):
        """
        判断是不是单探头
        :return:
        """
        # 整体转发时只有1个探头
        if self.meta['moduleCount'] == 1:
            return True
        else:
            # 单探头转发应该是2个探头，其中一个sn号为0
            if len(self.meta['modules'].keys()) == 2 and 0 in self.meta['modules'].keys():
                return True
            else:
                # 不满足以上两个条件就不是单人单探头
                return False

    def mergeMetaTriggerModule(self):
        """
        单探头转发时把trigger探头的信息合并到高速探头中
        :return:
        """
        # 通道数+1
        self.n_chan += 1
        # 其他附加上trigger探头的信息
        self.srates.extend(self.meta['modules'][0]['sampleRates'])
        self.channelNames.extend(self.meta['modules'][0]['channelNames'])
        self.channelTypes.extend(self.meta['modules'][0]['channelTypes'])
        self.maxDigital.extend(self.meta['modules'][0]['maxDigital'])
        self.minDigital.extend(self.meta['modules'][0]['minDigital'])
        self.maxPhysical.extend(self.meta['modules'][0]['maxPhysical'])
        self.minPhysical.extend(self.meta['modules'][0]['minPhysical'])
        self.gain.extend(self.meta['modules'][0]['gain'])
        self.dataCountPerChannel.extend(self.meta['modules'][0]['dataCountPerChannel'])

    def resolve(self, msg: bytes, isMeta: bool):
        """
        具体的解析数据的过程，协议见<数据转发协议-新版.xlsx>
        :return:
        """
        # 解析meta包
        if isMeta:
            if self.__hasMeta:
                # JellyFish收到这边发送的meta接收OK包之前可能还在发meta包
                # 已经解析过meta包就不要再解析了
                pass
            else:
                # 解析meta包
                self.meta = resolveMeta(msg)
                # 不是单探头就报错
                if not self.isSingleModule():
                    raise Exception('只能是单人单探头!')
                # 获取解析好的数据
                # 找到非0的SN号，能运行到这肯定存在非0的sn号
                for sn in self.meta['modules'].keys():
                    if sn != 0:
                        self.serialNumber = sn
                # 整体转发直接取信息即可
                self.n_chan = self.meta['modules'][self.serialNumber]['channelCount']
                # 防止单探头转发合并时把self.meta包给修改了，导致后面解析数据包出问题
                self.srates = self.meta['modules'][self.serialNumber]['sampleRates'].copy()
                self.channelNames = self.meta['modules'][self.serialNumber]['channelNames'].copy()
                self.channelTypes = self.meta['modules'][self.serialNumber]['channelTypes'].copy()
                self.maxDigital = self.meta['modules'][self.serialNumber]['maxDigital'].copy()
                self.minDigital = self.meta['modules'][self.serialNumber]['minDigital'].copy()
                self.maxPhysical = self.meta['modules'][self.serialNumber]['maxPhysical'].copy()
                self.minPhysical = self.meta['modules'][self.serialNumber]['minPhysical'].copy()
                self.gain = self.meta['modules'][self.serialNumber]['gain'].copy()
                self.dataCountPerChannel = self.meta['modules'][self.serialNumber]['dataCountPerChannel'].copy()
                self.personName = self.meta['modules'][self.serialNumber]['personName']
                self.moduleName = self.meta['modules'][self.serialNumber]['moduleName']
                self.moduleType = self.meta['modules'][self.serialNumber]['moduleType']
                if len(self.meta['modules'].keys()) == 2:
                    # 单探头转发把trigger的信息合并到高速探头中
                    self.mergeMetaTriggerModule()
                # 如果所有通道都是非EEG类型，报错退出
                if isChannelNotAllEEG(self.channelTypes):
                    raise Exception('所有通道都是非EEG类型!')
                # buffer长度
                nPoints = int(np.round(self.t_buffer * self.sample_rate))
                # 通道个数
                nChans = len(self.channelNames)
                # 初始化RingBuffer
                self.buffer = RingBuffer(nChans, nPoints)
                # DoubleBuffer可以存储数据，测试数据正确性
                self.save_buffer = DoubleBuffer(nChans, nPoints)
                # 发送 MetaData接收OK 确认包
                succ = bytes.fromhex('F55F5FF5')
                self.sock.send(succ)
                self.__hasMeta = True
        # 解析数据包
        else:
            if not self.__hasMeta:
                # meta包还没解析好就报错
                raise RuntimeError("Wrong program. Receive data before meta.")
            # 数据已经稳定了就认为准备好解析数据包了
            if self.state == ConnectState.CONNECTED and self.stabilizeCount >= self.pointsForStabilize:
                self.state = ConnectState.READY
                # 数据稳定后就不需要这个变量了,复位
                self.stabilizeCount = 0
            # 数据还没稳定
            elif self.state == ConnectState.CONNECTED:
                self.stabilizeCount += 1
                return
            # 数据包的解析结果
            dataStruct = resolveData(msg, self.meta)
            self.packet_count += 1
            # 整体转发
            if self.meta['flag'] % 2 == 0:
                self.isDataPacketLost(dataStruct)
                dataArr = dataStruct['modules'][self.serialNumber]['data']
                tempBuf = []
                for ch in range(self.n_chan):
                    tempBuf.append(dataArr[ch])
                # 重采样trigger通道
                tempBuf = self.ResampleTrigger(tempBuf)
                tempBuf = np.array(tempBuf)
                if self.state == ConnectState.RUNNING:
                    # 把数据添加到RingBuffer
                    self.buffer.appendBuffer(tempBuf)
                    # 也添加一份到DoubleBuffer，用于测试
                    self.save_buffer.appendBuffer(tempBuf)
                    # 时间戳
                    self.timeStamp['data'].append(dataStruct['startTimeStamp'])
                    # 获取Trigger信息
                    if self.__triggerLog is not None:
                        trgArr = np.asarray(dataArr[-1])
                        for idx, val in zip(np.where(trgArr>0)[0],
                                            trgArr[trgArr>0]):
                            tm: int = dataStruct["startTimeStamp"] + idx  # in ms
                            self.__triggerLog.write(f"{tm},{val}\n")
            # 按探头转发要组包
            else:
                sn = list(dataStruct['modules'].keys())[0]
                # trigger通道的sn号为0
                if sn == 0:
                    # 累计trigger的时间戳
                    self.timeStamp['trigger'].append(dataStruct['startTimeStamp'])
                    # 缓存trigger包
                    self.single_module_trigger_buffer.append(dataStruct)
                    # 输出收到的trigger包的信息
                    # self.trigger_count += 1
                    # print('trigger count:', self.trigger_count, 'trigger value:', dataStruct['modules'][0]['data'][0], 'time stamp:',
                    #       dataStruct['startTimeStamp'])
                else:
                    self.isDataPacketLost(dataStruct)
                    # 累计数据包的时间戳
                    self.timeStamp['data'].append(dataStruct['startTimeStamp'])
                    # 缓存数据包
                    self.single_module_data_buffer.append(dataStruct)
                # 缓存满了就进行组包操作
                if len(self.single_module_data_buffer) == self.max_single_packet:
                    self.combineDataAndTrigger()

    def ResampleTrigger(self, temBuf):
        """
        整体转发时非1000采样率时trigger通道和其他通道在一个包内的点数不同，需要重采样
        :param temBuf:一个包内各个通道的数据
        :return:
        """
        # 采样率为1000时不用做什么
        if self.sample_rate == 1000:
            return temBuf
        oldTriggerChannel = temBuf[-1]
        # 新trigger通道
        newTriggerChannel = [0] * len(temBuf[0])
        # 重采样比率
        rate = 1000 / self.sample_rate
        for i in range(len(oldTriggerChannel)):
            if oldTriggerChannel[i] > 0:
                newTriggerChannel[int(i / rate)] = oldTriggerChannel[i]
        temBuf[-1] = newTriggerChannel
        return temBuf

    def isDataPacketLost(self, dataStruct):
        """
        通过时间戳，验证是否数据包丢失
        :param dataStruct:解析好的数据包
        :return:
        """
        if self.state == ConnectState.RUNNING and self.firstTimestamp == -1:
            self.firstTimestamp = dataStruct["startTimeStamp"]
        # 通过时间戳验证是否丢包
        if self.lastTimestamp > 0 and self.lastTimestamp != dataStruct["startTimeStamp"]:
            raise RuntimeError(
                "Maybe a packet loss happened. Expected startTimestamp "
                f"is {self.lastTimestamp} but received "
                f"{dataStruct['startTimeStamp']}")
        self.lastTimestamp = dataStruct["startTimeStamp"] + dataStruct["timeStampLength"]

    def combineDataAndTrigger(self):
        """
        组包的具体过程
        :return:
        """
        # 所有通道的buffer，最后一个通道是trigger的数据
        temp_buffer = []
        for i in range(self.n_chan):
            temp_buffer.append([])
        # 把数据包放到前面通道的位置
        for i in range(self.max_single_packet):
            data = self.single_module_data_buffer[i]['modules'][self.serialNumber]['data']
            for ch in range(self.n_chan - 1):
                temp_buffer[ch].extend(data[ch])
        # temp_buffer最后一个通道初始化为0
        temp_buffer[-1] = [0] * len(temp_buffer[0])
        # 计算每包数据点dataCount，转发时长 * 采样率 / 1000
        dataCount = int(self.single_module_data_buffer[0]['timeStampLength'] * self.sample_rate / 1000)
        # 得到这批数据的时间戳
        totalTimestamp = []
        for i in range(self.max_single_packet):
            totalTimestamp.append(self.single_module_data_buffer[i]['startTimeStamp'])
        # 倒着遍历，可以删除那些组到数据包中的trigger包
        for i in range(len(self.single_module_trigger_buffer) - 1, -1, -1):
            startTimestamp = self.single_module_trigger_buffer[i]['startTimeStamp']
            # 找到这个trigger包的时间戳在数据包的所有时间戳中的位置
            index = self.FindTriggerTimeStampIndex(totalTimestamp, startTimestamp, dataCount)
            # 这个trigger的时间戳比这组数据中最大的时间戳还大，继续循环
            if index == -1:
                continue
            # 这个trigger的时间戳比这组数据中最小的时间戳还小，舍弃这个trigger
            elif index == -2:
                # 去除掉这个trigger
                self.single_module_trigger_buffer.pop(i)
            else:
                # 把trigger的值放到对应的位置
                temp_buffer[-1][index] = self.single_module_trigger_buffer[i]['modules'][0]['data'][0][0]
                # 去除掉这个trigger
                self.single_module_trigger_buffer.pop(i)
        temp_buffer = np.array(temp_buffer)
        # 把组好的包送入Buffer
        if self.state == ConnectState.RUNNING:
            self.buffer.appendBuffer(temp_buffer)
            # 也送一份到save_buffer里，验证正确性
            self.save_buffer.appendBuffer(temp_buffer)
        # 清空缓存的数据包
        self.single_module_data_buffer = []

    def FindTriggerTimeStampIndex(self, totalTimestamp, triggerTimeStamp, dataCount):
        """
        找到trigger的时间戳在这组数据中的位置
        :param totalTimestamp:这批数据包所有的时间戳
        :param triggerTimeStamp:当前Trigger的时间戳
        :param dataCount:每包数据点
        :return:
        """
        # currentTimeStamp比totalTimestamp中最大的时间戳还大，要保留
        if triggerTimeStamp > totalTimestamp[-1]:
            return -1
        # currentTimeStamp比totalTimestamp中最小的时间戳还小，要舍弃
        if triggerTimeStamp < totalTimestamp[0]:
            return -2
        # 不是以上两种情况就先找到离这个时间戳最近的数据包的时间戳的位置
        # 就是第几个数据包的位置
        properTimeStampIndex = -1
        for timeStamp in totalTimestamp:
            if timeStamp <= triggerTimeStamp:
                properTimeStampIndex += 1
            else:
                # 出现第一个大于trigger时间戳就说明找到了
                break
        # 如果找到的时间戳正好和trigger的时间戳相等
        # 就把这个时间戳直接放到对应的数据包的第一个位置
        if totalTimestamp[properTimeStampIndex] == triggerTimeStamp:
            index = properTimeStampIndex * dataCount
        else:
            # 如果找到的时间戳正好比trigger的时间戳小，就分采样率讨论
            # 采样率大于等于1000时，找到精确的时间戳的位置
            if self.sample_rate >= 1000:
                # 这个包内部计算时间戳精确位置
                subTimeStampIndex = (triggerTimeStamp - totalTimestamp[properTimeStampIndex]) * int(self.sample_rate / 1000)
                # 最终的位置就是包开始的位置+包内的位置
                index = properTimeStampIndex * dataCount + subTimeStampIndex
            else:
                # 采样率<1000时，就直接放到对应数据包的第一个位置
                index = properTimeStampIndex * dataCount
        return index

    def GetDataLenCount(self):
        """
        用于统计更新了多少数据
        :return:
        """
        return self.buffer.nUpdate

    def ResetDataLenCount(self):
        """
        重置已经更新的数据量
        :return:
        """
        self.buffer.nUpdate = 0

    def ResetTriggerChanofBuff(self):
        """
        把Trigger通道的值都置为0
        """
        self.buffer.buffer[-1, :] = np.zeros((1, self.buffer.buffer.shape[-1]))

    def GetBufferData(self):
        """
        获取buffer中的数据
        :return:
        """
        return self.buffer.getData()

    def getSaveDataBuffer(self):
        temBuf = self.save_buffer.getData()
        return temBuf

    def process_trig(self):
        trig = self.getSaveDataBuffer()[-1]
        assert trig.ndim == 1
        rst = []
        ids = np.where(trig > 0)[0]  # indices of triggers in
        # sample points
        for i in ids:
            trg = str(int(trig[i]))
            rst.append((i / self.sample_rate, 0, trg))  # (onset duration,desc)
        return rst

    def save(self, fpath: str):
        signal_headers = []
        for ich in range(self.n_chan - 1):
            signal_headers.append(
                highlevel.make_signal_header(
                    self.channelNames[ich],
                    dimension='uV',
                    sample_rate=self.srates[ich],
                    physical_min=self.minPhysical[ich],
                    physical_max=self.maxPhysical[ich],
                    digital_min=self.minDigital[ich],
                    digital_max=self.maxDigital[ich]))
        header = highlevel.make_header(patientname='s', gender='Unknown')
        header['annotations'] = self.process_trig()
        # [file type]
        # EDF = 0, EDF+ = 1, BDF = 2, BDF+ = 3, automatic from extension = -1
        highlevel.write_edf(fpath, self.getSaveDataBuffer()[:-1], signal_headers, header, file_type=3)

    def save_timeStamp(self):
        """
        保存时间戳
        :return:
        """
        with open('./test_time_stamp.pickle', 'wb') as f:
            pickle.dump(self.timeStamp, f)
