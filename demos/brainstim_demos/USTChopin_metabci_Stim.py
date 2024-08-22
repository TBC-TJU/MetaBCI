import time
import random
import pygame
from pygame.locals import *
#from pylsl import StreamInfo, StreamOutlet
import numpy as np
from datetime import datetime
import serial
import threading
import argparse
from psychopy import visual, core, event
#from metabci.brainflow.amplifiers import dataserver_thread
from metabci.brainstim.paradigm import PlayPiano


# 运行实验，设置实验重复次数、休息时间、准备时间、想象时间
def run_experiment(experiment, num_repeats=4, rest_time=2, prepare_time=2, imagine_time=5, onlinemode=False):
        for repeat in range(num_repeats):
            tasks = list(random.sample(range(1, 6), 5))
            for i, task in enumerate(tasks):
                if not experiment.exit_detection():
                    break
                experiment.display_text(texts=[f'KEY {task}','+'], colors=(1, 1, 1), positions=(0, experiment.win.size[1]//4))
                #self.display_text('+', color=(1, 1, 1), x=0, y=0)
                core.wait(prepare_time)
                
                experiment.display_text_image(text=f'KEY {task}', image_path=f'./demos/brainstim_demos/Stim_images/L{task}R{task}.jpg', color=(0, 0, 0), position=[(0, 0)])
                core.wait(imagine_time)
                data=experiment.collect_data(task, repeat, i, 1, onlinemode)
                
                experiment.display_text(texts='休息', colors=(1, 1, 1), positions=(0,0))
                core.wait(rest_time+2)
                
                data=experiment.collect_data(0, repeat, i, 0, onlinemode)
                #print(data)
                # #self.display_text_image(text=f'KEY {task}', image_path=f'./demos/brainstim_demos/Stim_images/L{task}R{task}.jpg', color=(1, 1, 1), position=[(0, 0)])
                # experiment.display_images_in_corners(task=task)
                # core.wait(imagine_time)
                
                # data=experiment.collect_data(task, repeat, i, 2, onlinemode)
                
                experiment.display_text(texts='休息', colors=(1, 1, 1), positions=(0,0))
                core.wait(rest_time)
                
                #self.collect_data(0, repeat, i, 0)
            
            experiment.display_text(texts='间期休息', colors=(1, 1, 1), positions=(0,0))
            core.wait(rest_time + 3)
        
        experiment.thread_data_server.stop()
        experiment.win.close()

experiment = PlayPiano(ip='127.0.0.1', port=8712, device_idx=0, mode ='offline',Time_buffer=5)
run_experiment(experiment, num_repeats=4, rest_time=2, prepare_time=2, imagine_time=5)
experiment.save_data("ShuK_0821_2")

