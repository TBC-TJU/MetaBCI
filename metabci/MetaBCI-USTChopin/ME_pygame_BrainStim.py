import time
import random
import pygame
from pygame.locals import *
#from pylsl import StreamInfo, StreamOutlet
from neuracle_lib.dataServer import dataserver_thread
import numpy as np
from datetime import datetime
import serial
from neuracle_lib.triggerBox import TriggerBox,TriggerIn,PackageSensorPara
import threading
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='127.0.0.1')# 192.168.31.236
    parser.add_argument('--port', type=str, default= 8712)
    args= parser.parse_args()
    return args
args = parse_args()
hostname = args.ip
port = args.port
print("args:",hostname,port)
# ip = '127.0.0.1'
# 58+1 channels;  与Neuracle recorder中的dataservice Montage顺序保持一致
# 其中30 EEG，28 sEMG, 1 TRG
neuracle = dict(device_name = 'Neuracle',hostname = hostname, port = int(port),
                srate = 1000,chanlocs = ['FP1','FP2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
                                            'CP5','CP1','CP2','CP6','P7','P3','Pz','P4','P8','PO3','PO4','O1','Oz','O2',
                                            'ECG','PO7','TP7','P5','FT7','FC3','F5','AF7','AF3','F1','C5','CP3',
                                            'POz','PO6','PO5','PO8','P6','TP8','C6','CP4','FT8','FC4','F6','AF8','F2','FCz','AF4','FPz','TRG'],
                                            n_chan = 59)
# neuracle = dict(device_name = 'Neuracle',hostname = '127.0.0.1',port = 8712,
#                 srate = 1000,chanlocs = ['FP1'(0),'FP2'(1),'F7'(2),'F3'(3),'Fz'(4),'F4'(5),'F8'(6),'FC5'(7),'FC1'(8),'FC2'(9),'FC6'(10),'T7'(11),'C3'(12),'Cz'(13),'C4'(14),'T8'(15),
#                                             'CP5'(16),'CP1'(17),'CP2'(18),'CP6'(19),'P7'(20),'P3'(21),'Pz'(22),'P4'(23),'P8'(24),'PO3'(25),'PO4'(26),'O1'(27),'Oz'(28),'O2'(29),
#                                             'ECG'(30),'PO7'(31),'TP7'(32),'P5'(33),'FT7'(34),'FC3'(35),'F5'(36),'AF7'(37),'AF3'(38),'F1'(39),'C5'(40),'CP3'(41),
#                                             'POz'(42),'PO6'(43),'PO5'(44),'PO8'(45),'P6'(46),'TP8'(47),'C6'(48),'CP4'(49),'FT8'(50),'FC4'(51),'F6'(52),'AF8'(53),'F2'(54),'FCz'(55),'AF4'(56),'FPz'(57),'TRG'(58)],
#                                             n_chan = 59)

# neuracle = dict(device_name = 'Neuracle',hostname = '127.0.0.1',port = 8712,
#                 srate = 1000,chanlocs = ['FPz','AF3'],n_chan = 2)

dsi = dict(device_name = 'DSI-24',hostname = hostname,port = port,
            srate = 300,chanlocs = ['P3','C3','F3','Fz','F4','C4','P4','Cz','CM','A1','Fp1','Fp2','T3','T5','O1','O2','X3','X2','F7','F8','X1','A2','T6','T4','TRG'],n_chan = 25)

device = [neuracle,dsi]

# s_com = serial.Serial('COM6', 115200, timeout = None) #com6 替换为实验主机的端口号（详见硬件安装）

### pay attention to the device you used
target_device = device[0]
srate = target_device['srate']
print('!!!! The type of device you used is %s'%target_device['device_name'])

## init dataserver
time_buffer = 5 # second
thread_data_server = dataserver_thread(threadName='data_server', device=target_device['device_name'], n_chan=target_device['n_chan'],
                                        hostname=target_device['hostname'], port= target_device['port'],srate=target_device['srate'],t_buffer=time_buffer)
thread_data_server.Daemon = True
notconnect = thread_data_server.connect()
if notconnect:
    raise TypeError("Can't connect recorder, Please open the hostport ")
else:
    thread_data_server.start()
    print('Data server connected')

    
 
# 创建一个LSL流来发送实验标记
#info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw43536')
#outlet = StreamOutlet(info)

# 初始化 Pygame,获取显示器信息,并获取当前显示器的分辨率
pygame.init()
screen_info = pygame.display.Info()
current_width = screen_info.current_w
current_height = screen_info.current_h
  
# 设置屏幕大小和字体
# screen_width = 800
# screen_height = 600
# screen = pygame.display.set_mode((screen_width, screen_height))
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

pygame.display.set_caption('MetaBCI_BrainStim_USTChopin')
font = pygame.font.Font(None, 200)
font_size = 74
font_Chinese = pygame.font.Font('C:/Windows/Fonts/STKAITI.ttf', font_size)  # STKAITI华文正楷；none使用默认字体
small_font = pygame.font.Font(None, 36)

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# 图片地址
# path = 'D:/USTC/Program/metaBCI/MetaBCI-master/MetaBCI-master/metabci/brainstim/textures'
path = './Stim_images/'
image_cross_large = path+'/cross-large.png'  # 替换为你的图片路径
image_L0R0 = path+'/L0R0.jpg'    # 替换为你的图片路径
image_L1R1 = path+'/L1R1.jpg'    # 替换为你的图片路径
image_L2R2 = path+'/L2R2.jpg'    # 替换为你的图片路径
image_L3R3 = path+'/L3R3.jpg'    # 替换为你的图片路径
image_L4R4 = path+'/L4R4.jpg'    # 替换为你的图片路径
image_L5R5 = path+'/L5R5.jpg'    # 替换为你的图片路径

# 定义显示文本函数
def display_text(text, fonts, color, screen, x, y):
    text_surface = fonts.render(text, True, color)
    rect = text_surface.get_rect()
    rect.center = (x, y)
    screen.blit(text_surface, rect)
    pygame.display.flip()

# 定义显示图片函数
def display_image(image_path, color, screen):
    # 填充背景色
    screen.fill(color)

    # 获得图片及其尺寸
    image = pygame.image.load(image_path)
    image_width, image_height = image.get_size()

    # 将图片居中显示
    x = (current_width - image_width) // 2
    y = (current_height - image_height) // 2 

    # 将图片绘制到窗口上
    screen.blit(image, (x, y))
    pygame.display.flip()

# 定义显示图片函数
def display_text_image(text,fonts,image_path, color, screen):
    # 填充背景色
    screen.fill(color)

    # 获得图片及其尺寸
    image = pygame.image.load(path+image_path+'.jpg')
    image_width, image_height = image.get_size()

    # 将图片居中显示
    x = (current_width - image_width) // 2
    y = (current_height - image_height) // 2 
    # 将图片绘制到窗口上
    screen.blit(image, (x, y+100))
 
    #显示提示词
    text_surface = fonts.render(text, True, BLACK)
    rect = text_surface.get_rect()
    rect.center = (x+500, y-10) #左上角为原点，向右为x正方向，向下为y轴正方向
    screen.blit(text_surface, rect)   

    #更新显示
    pygame.display.flip()

# 设置实验参数
num_repeats = 6  # 总试次数 = num_repeats * trials
rest_time = 2    # 休息时间（秒）
prepare_time = 2  # 休息时间（秒）
imagine_time = 5  # 想象时间（秒）

# 初始化实验数据记录
experiment_data = []

# 定义第一个线程的函数
def thread1_exit_detection():
     # 检查退出事件
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
            break
        # 检测键盘按键事件
        elif event.type == pygame.KEYDOWN:  #有键被按下
            if event.key == pygame.K_SPACE: #空格键被按下 
                running = False   
                break 
    return running==True


#主实验循环
for repeat in range(num_repeats):
    # 随机选择一个想象动作
    tasks =  list(random.sample(range(1, 6), 5))
    num_task = len(tasks)
    i = 0
    
    running = True
    for task in tasks:
        i = i+ 1
        # 检查退出事件
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                break
            # 检测键盘按键事件
            elif event.type == pygame.KEYDOWN:  #有键被按下
                if event.key == pygame.K_SPACE: #空格键被按下 
                    running = False   
                    break             
        if running==False:
            break
        # 显示固定点，提示被试准备
        screen.fill(BLACK)
        display_text('KEY'+" " +str(task), font, WHITE, screen, current_width//2, current_height//2-150)
        display_text('+', font, WHITE, screen, current_width//2, current_height//2)
        time.sleep(prepare_time)
        
        # 显示想象指令，并发送相应的标记
        screen.fill(BLACK)
        image = 'L'+str(task)+'R'+str(task)
        display_text_image('KEY'+" " +str(task),font_Chinese,image, WHITE, screen)
        marker = task    
        # s.write([task])
        # outlet.push_sample([task])
        # start_time = time.time()  # 记录开始时间
        time.sleep(imagine_time)
     
        flagstop = False
        try:
            while not flagstop: # get data in one second step
                # s_com.write([task])
                data = thread_data_server.get_bufferData()
                flagstop = True
        except:
            pass

        # 记录结束标记
        # outlet.push_sample([0])
        # end_time = time.time()  # 记录结束时间
        
        # 保存实验数据
        experiment_data.append({
            'data': data,
            'trial': repeat*num_task + i,
            'marker': task
        })
    
        # 确保被试有足够的休息时间
        screen.fill(BLACK)
        display_text('休息', font_Chinese, WHITE, screen, current_width//2, current_height//2)

        time.sleep(rest_time) 
        flagstop = False
 
        try:
            while not flagstop: # get data in one second step   
                # s_com.write([6])
                data = thread_data_server.get_bufferData()[:,-800:]
                flagstop = True
        except:
            pass
        # 保存实验数据
        experiment_data.append({
            'data': data,
            'trial': repeat*num_task + i,
            'marker': 0
        })
    # 确保被试在每一轮动作后有足够的休息时间

    screen.fill(BLACK)
    display_text('间期休息', font_Chinese, WHITE, screen, current_width//2, current_height//2)
    time.sleep(rest_time+3) 

thread_data_server.stop()

# 结束实验
pygame.quit()


# 获取当前时间
now = datetime.now()

# 格式化时间输出为"年-月-日 时:分"
Exp_date = now.strftime("%Y_%m_%d %H_%M")
Subject = 'LuHK'
print("当前时间：", Exp_date)#D:/USTC/Program/metaBCI
np.save('./'+Subject +'_'+Exp_date+'_data.npy',experiment_data)
