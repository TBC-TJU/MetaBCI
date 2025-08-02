import win32file
import win32pipe
import struct
import threading
import time
import pyautogui
import sys
# from psychopy import parallel
import serial
from metabci.brainstim.utils import NeuraclePort

import warnings
import numpy as np
import joblib
import socket

sys.path.append('..')
Flag = '\0'
endflag = False
warnings.filterwarnings('ignore')  # or warnings.filterwarnings("default")


# 串口地址
def set_serial(p_port):
    global port
    if (0 != p_port):
         port = NeuraclePort(port_addr=p_port)
    else:
         port = 0



# bind udp
def set_socket_udp(ip, port, timeout):
    global udp
    socket.setdefaulttimeout(timeout)
    udp = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    server_ip = ip
    server_port = port
    udp.bind((server_ip, server_port))

# 标签置零
def set_zero():     # 改成博睿康打标
    if (0 != port):
        port.setData(0)

# 标签
def set_marker():
    if (0 != port):
        if Flag == '1':
            port.setData(1)  # Flag=1时发送标签1
        elif Flag == '2':
            port.setData(2)  # Flag=2时发送标签2
        else:
            port.setData(1)  # 其他情况默认发送标签1

def udp_recv():
    try:
        # 接收客户端发来的数据，包括bytes对象data，以及客户端的IP地址和端口号addr，其中addr为二元组(host, port)。
        tar_index, address = udp.recvfrom(1)
        print("Received:", tar_index, "from", address)
        tar_char = tar_index[0]  # 真实结果
        # tar_char = random.randint(1, 9)  # 随机结果 1-9
        level = int(tar_char);
        return level
        # print('*****************************************: ' + str(int(win)))
    except:
        # print('nothing detected,continue to next run')
        print('')
        return 0

### 根据信号等级响应两种任务，患侧转弯（QWERT）和刹车（ASDFG）
def command_controller(level):
    if level != 0:
        if Flag == '1':
            if level == 1:
                pyautogui.press("q")
            elif level == 2:
                pyautogui.press("w")
            elif level == 3:
                pyautogui.press("e")
            elif level == 4:
                pyautogui.press("r")
            elif level == 5:
                pyautogui.press("t")
        elif Flag == '2':
            if level == 1:
                pyautogui.press("y")
            elif level == 2:
                pyautogui.press("u")
            elif level == 3:
                pyautogui.press("i")
            elif level == 4:
                pyautogui.press("o")
            elif level == 5:
                pyautogui.press("p")
        elif Flag == '3':
            if level == 1:
                pyautogui.press("a")
            elif level == 2:
                pyautogui.press("s")
            elif level == 3:
                pyautogui.press("d")
            elif level == 4:
                pyautogui.press("f")
            elif level == 5:
                pyautogui.press("g")


### 获取EEG处理结果，并反馈信号等级（
def event_progress():
    print("running")
    T0 = time.time()
    set_marker()
    print("Time", T0)
    T1 = time.time()
    set_zero()
    targ = 0
    count = 1
    result = 0
    while Flag != '0':
        T1 = time.time()
        # if T1-T0>count*0.7:
        #     set_marker()
        #     set_zero()
        #     count+=1

        print("Ready Time", T1)
        #set_zero()
        if (T1-T0)>=4:
            result = udp_recv()
            if result != 0 :
                print("Receive Data", result)
        #targ +=1
        if result == 1 :
            if Flag =="1" or Flag == "3":
                level = 5
            else:
                level = 1
            command_controller(level)
        elif result == 2 :
            if Flag =="2" or Flag == "4":
                level = 5
            else:
                level = 1
            command_controller(level)
        T2 = time.time()
        # while T2 - T1 < 0.7:
        #     T2 = time.time()
    print("Over Time", T2)
    # while targ != count:
    #     udp_recv()
    #     targ+=1



## 管道通信
def recv_pipe(PIPE_NAME, PIPE_BUFFER_SIZE):
    global endflag
    global udp
    # 36 cue & stim
    while not endflag:
        named_pipe = win32pipe.CreateNamedPipe(PIPE_NAME,
                                               win32pipe.PIPE_ACCESS_DUPLEX,
                                               win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_WAIT | win32pipe.PIPE_READMODE_MESSAGE,
                                               win32pipe.PIPE_UNLIMITED_INSTANCES,
                                               PIPE_BUFFER_SIZE,
                                               PIPE_BUFFER_SIZE, 500, None)
        try:
            while not endflag:
                try:
                    win32pipe.ConnectNamedPipe(named_pipe, None)
                    data = win32file.ReadFile(named_pipe, PIPE_BUFFER_SIZE, None)
                    if data is None:
                        continue
                    # print("Received msg:", data)
                    recv_msg = struct.unpack('<2s', data[1])
                    recv_msg = recv_msg[0].decode("utf-8")
                    global Flag
                    Flag = recv_msg[0]
                    print("Flag:", Flag)
                    task_event = threading.Event()
                    if Flag == 'e':
                        # read_data.stop_acq()
                        print("Data read is stop")
                        endflag = True
                    elif Flag == 'b':
                        # 执行采集
                        print("Data is reading")
                        Flag = '\0'
                    elif Flag != '0':
                        control_thread = threading.Thread(target=event_progress)
                        control_thread.start()
                    else:
                        task_event.clear()
                    print("Parsed message:", recv_msg)
                except BaseException as e:
                    print("Exception1:", e)
                    break
        finally:
            try:
                win32pipe.DisconnectNamedPipe(named_pipe)
            except BaseException as e:
                print("Exception2:", e)
                break

if __name__ == '__main__':
    # 初始化管道、读取、处理
    pipe_name = r"\\.\pipe\test_pipe"
    pipe_buffer_size = 65535
    port_addr = 'COM5'
    # set_parallel(0x3EFC)  #206
    # set_socket_udp('192.168.0.103', 9094, 0.01)  # 206
    set_serial(port_addr)  # 206     # 改成串口打标签
    #set_serial('COM5')
    # set_socket_udp('192.168.1.103', 9094, 0.01)  # 206
    set_socket_udp('192.168.1.102', 9095, 0.01)
    # set_parallel(0)
    # set_socket_udp('169.254.29.63', 9094, 0.01) # 107
    event_thread_process = threading.Event()
    recv_pipe(pipe_name, pipe_buffer_size)
    print("All is over.")
    time.sleep(5)

# if __name__ == '__main__':
#     # 初始化管道、读取、处理
#     pipe_name = r"\\.\pipe\test_pipe"
#     pipe_buffer_size = 65535
#     port_addr = 'COM5'
#
#     set_serial(port_addr)
#     set_socket_udp('192.168.1.102', 9095, 10)
#
#
#     # 添加UDP数据接收监听线程
#     def udp_listener():
#         while True:
#             try:
#                 data, addr = udp.recvfrom(1024)
#                 print("\n[UDP] Received data:", data, "from", addr)
#                 print("[UDP] Data content:", data.decode('utf-8', errors='replace'))  # 尝试解码内容
#             except Exception as e:
#                 print("[UDP] Error receiving data:", e)
#
#
#     # 启动UDP监听线程
#     udp_thread = threading.Thread(target=udp_listener, daemon=True)
#     udp_thread.start()
#
#     print("System initialized. Waiting for data...")
#     event_thread_process = threading.Event()
#     recv_pipe(pipe_name, pipe_buffer_size)
#     print("All is over.")
#     time.sleep(5)