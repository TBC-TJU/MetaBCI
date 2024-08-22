import os
import os.path as op
import string
import numpy as np
from math import pi
from psychopy import data, visual, event,monitors
import datetime
import socket
import struct
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List, Optional, Tuple, Dict, Any
import random
import numpy as np


from  threading import Lock, Thread, Event,Timer
import select,time
from struct import unpack
class ringBuffer():
    def __init__(self,n_chan,n_points):
        self.n_chan = n_chan
        self.n_points = n_points
        self.buffer = np.zeros((n_chan, n_points))
        self.currentPtr = 0
        self.nUpdate = 0
    ## append buffer and update current pointer
    def appendBuffer(self,data):
        n = data.shape[1]
        self.buffer[:,np.mod(np.arange(self.currentPtr,self.currentPtr+n),self.n_points)] = data
        self.currentPtr =  np.mod(self.currentPtr+n-1, self.n_points) + 1
        self.nUpdate = self.nUpdate+n
    ## get data from buffer
    def getData(self):
        data = np.hstack([self.buffer[:,self.currentPtr:], self.buffer[:,:self.currentPtr]])
        return data
    # reset buffer
    def resetBuffer(self):
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.currentPtr = 0
        self.nUpdate = 0
        
class NeuracleServerThread(Thread,):
    def __init__(self,threadName,device,n_chan, hostname='127.0.0.1', port= 8712,srate=1000,t_buffer=3):
        Thread.__init__(self)
        self.name = threadName
        self.sock = []
        self.device = device
        self.n_chan = n_chan
        self.hostname = hostname
        self.port = port
        self.t_buffer = t_buffer
        self.srate = srate
        self._update_interval = 0.04 ## unit is seconds. dataserver sends TCP/IP socket in 40 milliseconds


    def connect(self):
        """
        try to connect data server
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        notconnect = True
        reconnecttime = 0
        while notconnect:
            try:
                self.sock.connect((self.hostname, self.port))
                notconnect = False
            except:
                reconnecttime += 1
                print('connection failed, retrying for %d times' % reconnecttime)
                time.sleep(1)
                if reconnecttime > 2:
                    break
        self.shutdown_flag = Event()
        self.shutdown_flag.set()
        self.sock.setblocking(True)
        self.bufsize = int(self._update_interval*4*self.n_chan*self.srate*10)  # set buffer size
        self.npoints_buffer = int(np.round(self.t_buffer*self.srate))
        self.ringBuffer = ringBuffer(self.n_chan, self.npoints_buffer) # initiate the ringbuffer class
        self.buffer = b'' ## binary buffer used to collect binary array from data server
        return notconnect

    def run(self):
        self.read_thread()

    def read_thread(self): ## visit dataserver, catch sockets and parse sockets, append parsed data to ringbuffer
        socket_lock = Lock()
        while self.shutdown_flag.isSet():
            if not self.sock:
                break
            rs, _, _ = select.select([self.sock], [], [], 9)
            for r in rs:
                socket_lock.acquire()
                if not self.sock:
                    socket_lock.release()
                    break
                try:
                    raw = r.recv(self.bufsize)
                except:
                    print('can not recieve socket ...')
                    socket_lock.release()
                    self.sock.close()
                else:
                    raw = self.buffer + raw
                    data, evt = self.parseData(raw) ## parse data
                    socket_lock.release()
                    data = data.reshape(len(data) // (self.n_chan), self.n_chan)
                    self.ringBuffer.appendBuffer(data.T)
                    # print(self.ringBuffer.nUpdate)

                    # if len(data) > 0:
                    #     data = data.reshape(len(data) // (self.n_chan), self.n_chan)
                    #     self.ringBuffer.appendBuffer(data.T)
                    #     # print(self.ringBuffer.nUpdate)

                    # else:
                    #     print('Server Closed')
                    #     self.connect() # try connect again

    def parseData(self,raw):
        if  'Neuracle' in self.device: ## parsa data according to Neuracle device protocol
            n = len(raw)
            event , hexData  = [], []
            hexData = raw[:n - np.mod(n, 4 * self.n_chan)] # unpack hex-data  in row
            self.buffer = raw[n - np.mod(n, 4 * self.n_chan):]
            n_item = int(len(hexData)/4/self.n_chan)
            # format_str = '<' + (str(self.n_chan -1) + 'f' + '1I') * n_item
            format_str = '<' + (str(self.n_chan) + 'f') * n_item
            parse_data = unpack(format_str, hexData)

        elif  'DSI' in self.device : ## parsa data according to DSI device protocol
            token = '@ABCD'
            n = len(raw)
            i = 0
            parse_data, data_record, event, event_record  = [], [], [], []
            iData = 0
            iEvent = 1
            while i + 12 < n:
                if token == raw[i:i + 5].decode('ascii'):
                    packetType = raw[i + 5]
                    # print(packetType)
                    bytenum = raw[i + 6:i + 8]
                    packetLength = 256 * bytenum[0] + bytenum[1]
                    # bytenum = unpack('>4I', buffer[i+8:i+12])
                    # packetNumber = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                    if i + 12 + packetLength > n:
                        break
                    if packetType == 1:
                        data_record.append({})
                        # bytenum = unpack('>4I', buffer[i+12:i+16])
                        # data_record[iData]['TimeStamp'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        # data_record[iData]['DataCounter'] = unpack('>I', buffer[i+16])
                        # data_record[iData]['ADCStatus'] = unpack('>I', buffer[i+17:i+23])[0]
                        if np.mod(packetLength - 11, 4) != 0:
                            print('The packetLength may be incorrect!')
                        else:
                            pass
                        data_num = int((packetLength - 11) / 4)
                        format = '>' + str(data_num) + 'f'
                        data_record[iData]['ChannelData'] = unpack(format, raw[i + 23:i + 12 + packetLength])
                        parse_data.extend(data_record[iData]['ChannelData'])
                        iData += 1
                    elif packetType == 5:
                        event_record.append({})
                        # bytenum = unpack('>4I', buffer[i+12:i+16])
                        # event_record[iEvent]['EventCode'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        # bytenum = unpack('>4I', buffer[i+16:i+20])
                        # event_record[iEvent]['SendingNode'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        # if packerLength > 20:
                        #     bytenum = unpack('>4I', buffer[i+20:i+24])
                        #     event_record[iEvent]['MessageLength'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        #     event_record[iEvent]['Message'] = buffer[i+24:i+24+event[iEvent]['MessageLength']].decode('ascii')
                        # event.extend(event_record[iEvent]['Message'])
                        iEvent += 1
                    else:
                        pass
                    i = i + 12 + packetLength
                else:
                    i += 1
            self.buffer = raw[i:]
        else:
            print('not avaliable device !')
            parse_data =[]
            event = []
            pass
        return np.asarray(parse_data), event


    ## get float data
    def get_bufferData(self):
        return self.ringBuffer.getData()

    # get current update point
    def get_bufferNupdate(self):
        return  self.ringBuffer.nUpdate

    # set current update point
    def set_bufferNupdate(self,nUpdate):
        self.ringBuffer.nUpdate = nUpdate

    # reset current update point
    def ResetDataLenCount(self, count=0):
        self.ringBuffer.nUpdate = count

    # get current update point
    def GetDataLenCount(self):
        return self.ringBuffer.nUpdate

    # reset buffer
    # get current update point
    def resetBuffer(self):
        return self.ringBuffer.resetBuffer()
   

    # stop/close thread
    def stop(self):
        self.shutdown_flag.clear()
        
        
class USTChopin:
    def __init__(self, ip='127.0.0.1', port=8712, device_idx=0, mode='offline',Time_buffer=5):
        self.t_buffer = Time_buffer
        self.devices = [
            dict(device_name='Neuracle', hostname=ip, port=port, srate=1000,
                 chanlocs=['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                           'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3', 'PO4', 'O1', 'Oz', 'O2',
                           'ECG', 'PO7', 'TP7', 'P5', 'FT7', 'FC3', 'F5', 'AF7', 'AF3', 'F1', 'C5', 'CP3',
                           'POz', 'PO6', 'PO5', 'PO8', 'P6', 'TP8', 'C6', 'CP4', 'FT8', 'FC4', 'F6', 'AF8', 'F2', 'FCz', 'AF4', 'FPz', 'TRG'],
                 n_chan=59),
            dict(device_name='DSI-24', hostname=ip, port=port, srate=300,
                 chanlocs=['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'CM', 'A1', 'Fp1', 'Fp2', 'T3', 'T5', 'O1', 'O2', 'X3', 'X2', 'F7', 'F8', 'X1', 'A2', 'T6', 'T4', 'TRG'],
                 n_chan=25)
        ]
        self.device = self.devices[device_idx]
        self.srate = self.device['srate']
        
        self.thread_data_server = None
        self.experiment_data = []
        
        if mode=='offline':
            self.init_dataserver()
            self.init_psychord()
        elif mode=='online':
            self.init_dataserver()
            self.init_notation()
        else:
            print('The data type of the variable mode is incorrect!')
    
    def init_notation(self):
        full_screen_win = visual.Window(fullscr=True, monitor="testMonitor")
        screen_width, screen_height = full_screen_win.size
        full_screen_win.close()
        # monitor = monitors.Monitor('notation')
        self.screen_size = (screen_width,screen_height)#monitor.getSizePix()
        self.win = visual.Window(
            size=(self.screen_size[0]//2.2, self.screen_size[1]//1.7),  # 窗口大小
            pos=(self.screen_size[0]//4-self.screen_size[0]//5.6,self.screen_size[1]//1.26-self.screen_size[1]//1.7),  # 窗口位置 (x, y)
            color='black',  # 背景颜色
            units='pix'     # 单位设置为像素
        )
        self.screen_size = self.win.size

    def init_psychord(self):
        # 初始化psychopy的窗口
        self.win = visual.Window(fullscr=True, color=(0, 0, 0), units='pix')
        self.font_size = 150
        self.font = 'Arial'
        self.text_stim = visual.TextStim(win=self.win, font=self.font, height=self.font_size, color=(0, 0, 0))
        self.screen_size = self.win.size
    def init_dataserver(self):
        time_buffer = self.t_buffer  # second
        self.thread_data_server = NeuracleServerThread(
            threadName='data_server',
            device=self.device['device_name'],
            n_chan=self.device['n_chan'],
            hostname=self.device['hostname'],
            port=self.device['port'],
            srate=self.device['srate'],
            t_buffer=time_buffer
        )
        self.thread_data_server.Daemon = True
        notconnect = self.thread_data_server.connect()
        if notconnect:
            #raise TypeError("Can't connect recorder, Please open the hostport ")
            print(0)
        else:
            self.thread_data_server.start()
            print('Data server connected')

    def display_text(self, texts=[''], colors=[(0, 0, 0)], positions=[(0, 0)]):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(colors[0], float) or isinstance(colors[0], int):
            colors = [colors]
        if isinstance(positions[0], float) or isinstance(positions[0], int):
            positions = [positions]

        for text, color, position in zip(texts, colors, positions):
            text_stim = visual.TextStim(self.win, text=text, color=color,height=74, pos=position)
            text_stim.draw()
        self.win.flip()
    
    def display_image(self, image_path):
        image_stim = visual.ImageStim(self.win, image=image_path, size=self.win.size)
        image_width, image_height = image_stim.size
        scale_factor = min(self.screen_size[0] / image_width, self.screen_size[1] / image_height)
        image_stim.size = (image_width * scale_factor, image_height * scale_factor)
        image_stim.pos=(0,0)
        image_stim.draw()
        self.win.flip()
    
    def display_text_image(self, text, image_path, color=(0, 0, 0), position=[(0, 0)]):
        image_stim = visual.ImageStim(self.win, image=image_path)
        
        image_width, image_height = image_stim.size
        
        scale_factor = min(self.screen_size[0] / image_width, self.screen_size[1] / image_height)
        image_stim.size = (image_width * scale_factor, image_height * scale_factor)
        #print(scale_factor,image_stim.size)
        image_stim.draw()
        
        # 显示文字
        text_stim = visual.TextStim(self.win, text=text, color=color, height=74,pos=position)
        text_stim.draw()

        # 更新显示
        self.win.flip()

        # 持续一段时间
        #core.wait(2)  # 这里的2秒可以根据需要调整
    def display_images_in_corners(self):
        # 加载图像
        
        images = [f'./demos/brainstim_demos/Stim_images/{i}.jpg' for i in range(1, 5)]
        image_path = f'./demos/brainstim_demos/Stim_images/attention{1}.jpg'
        selected_images = random.sample(images, 2)
        positions = [
                     (-self.win.size[0] // 2 + 100, self.win.size[1] // 2 - 100),
                     (self.win.size[0] // 2 - 100, self.win.size[1] // 2 - 100),
                     (-self.win.size[0] // 2 + 100, -self.win.size[1] // 2 + 100),
                     (self.win.size[0] // 2 - 100, -self.win.size[1] // 2 + 100)]
        # 显示图像
        selected_positions = random.sample(positions, 2)
        for image, pos in zip(selected_images, selected_positions):
            # experiment.display_text_image(text=f'KEY {1}', image_path=f'./Stim_imgs/attention{1}.jpg', color=(0, 0, 0), position=[(0, 0)])
            img_stim1 = visual.ImageStim(win=self.win, image=image_path, pos=[(0, 0)])
            image_width, image_height = img_stim1.size 
            img_stim1.size = (image_width * 0.3, image_height * 0.3)

            img_stim = visual.ImageStim(win=self.win, image=image, pos=pos)
            image_width, image_height = img_stim.size 
            
            scale_factor = min(self.screen_size[0] / image_width, self.screen_size[1] / image_height)
            scale_factor /= 3
            img_stim.size = (image_width * scale_factor, image_height * scale_factor)
            img_stim1.draw()
            img_stim.draw()
        
        self.win.flip()
        
    def exit_detection(self):
        if 'space' in event.getKeys():
            return False
        return True
    
    def collect_data(self, task, repeat, i, marker2, online=False):
        flagstop = False
        data = None
        while not flagstop:
            try:
                data = self.thread_data_server.get_bufferData()
                flagstop = True
            except:
                pass

        self.experiment_data.append({
            'data': data,
            'trial': repeat * len(self.experiment_data) + i + 1,
            'marker1': task,
            'marker2': marker2
        })
        if online:
            return data
        else:
            return None
    def save_data(self, subject='Subject'):
        # now = datetime.now()
        # exp_date = now.strftime("%Y_%m_%d %H_%M")
        np.save(f'./{subject}_data.npy', self.experiment_data)