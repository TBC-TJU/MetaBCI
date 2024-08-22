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
def run_experiment(experiment, num_repeats=10, rest_time=2, prepare_time=2, imagine_time=5, onlinemode=False, attention=True):
        for repeat in range(num_repeats):
                core.wait(prepare_time)
                for i in range(30):
                        # tasks = list(random.sample(range(1, 3), 2))
                        tasks = list([1,2])
                        for j, task in enumerate(tasks):
                                if not experiment.exit_detection():
                                        break
                        # experiment.display_text(texts=[f'KEY {task}','+'], colors=(1, 1, 1), positions=(0, experiment.win.size[1]//4))
                        # #self.display_text('+', color=(1, 1, 1), x=0, y=0)
        
                        #data=experiment.collect_data(task, repeat, i, 1, onlinemode)
                        
                                experiment.display_text_image(text=f'KEY {task}', image_path=f'./demos/brainstim_demos/Stim_images/attention{task}.jpg', color=(0, 0, 0), position=[(0, 0)])
                                core.wait(imagine_time/30)
                        
                data=experiment.collect_data(task, repeat, 0, onlinemode)
                experiment.display_text(texts='间期休息', colors=(1, 1, 1), positions=(0,0))
                core.wait(rest_time + 3)

        for repeat in range(60):
                task = 0
                i = 0
                if not experiment.exit_detection():
                    break
                # experiment.display_text(texts=[f'KEY {task}','+'], colors=(1, 1, 1), positions=(0, experiment.win.size[1]//4))
                # #self.display_text('+', color=(1, 1, 1), x=0, y=0)
                # experiment.display_text(texts='准备', colors=(1, 1, 1), positions=(0,0))
                # core.wait(prepare_time)
                
                #data=experiment.collect_data(task, repeat, i, 1, onlinemode     
                experiment.display_images_in_corners()
                core.wait(imagine_time/30)
        data=experiment.collect_data(task, repeat, 1, onlinemode)
        experiment.display_text(texts='休息', colors=(1, 1, 1), positions=(0,0))
        core.wait(rest_time)
        i += 1
                
                
        experiment.thread_data_server.stop()
        experiment.win.close()
        
        
total_time = 60
num_repeats = 1

epoch_time = total_time / num_repeats
experiment = PlayPiano(ip='127.0.0.1', port=8712, device_idx=0, mode ='online',Time_buffer=60)
run_experiment(experiment, num_repeats=num_repeats, rest_time=1, prepare_time=1, imagine_time=epoch_time/2,attention=True)
experiment.save_data("Train_SHUK_0822_1")

experiment1 = PlayPiano(ip='127.0.0.1', port=8712, device_idx=0, mode ='online',Time_buffer=60)
run_experiment(experiment1, num_repeats=num_repeats, rest_time=1, prepare_time=1, imagine_time=epoch_time/2,attention=True)
experiment.save_data("Test_SHUK_0822_1")


