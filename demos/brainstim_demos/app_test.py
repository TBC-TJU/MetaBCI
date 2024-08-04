import multiprocessing
import sys
from abc import abstractmethod
from typing import Optional, Any
import os
from multiprocessing import Process, Lock ,Event, Queue, Manager
import queue
import time
from save_file_test import save_flie
import pyglet
from pyglet.window import mouse, key
import ctypes
import datetime
import gc
from collections import OrderedDict
from functools import partial
import numpy as np
from psychopy import core, visual, event, logging
#from .utils import _check_array_like, _clean_dict
import math
import time
from psychopy import monitors
import numpy as np

from psychopy.tools.monitorunittools import deg2pix
import pickle

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox


class APP(Process, pyglet.window.Window):
    def __init__(self, dict = {} , timeout: float = 5, name: Optional[str] = None,
                 w: int = 1920, h: int = 1080, screen_id: int = 0, *args, **kwargs):
        Process.__init__(self)
        self.running = Event()
        self.running.clear()
        self.pause = Event()
        self.pause.clear()
        self.handlers_exist = Event()

        self.timeout = timeout
        self.lock = Lock()
        self.App_name = name
        self._buffer = dict
        self.w = w
        self.h = h
        self.win_size = [self.w, self.h]
        self.screen_id = screen_id
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def main_App(self):
        pass

    def get_win(self, win_style='transparent', reload=False, screen_id=0, bg_color_warm=None):

        if reload:
            self.win_style = win_style
            # self.screen_id = screen_id
            # self.bg_color_warm = bg_color_warm
        ctypes.windll.user32.SetProcessDPIAware()
        _default_display_ = pyglet.canvas.get_display()
        allScrs = _default_display_.get_screens()
        thisScreen = allScrs[self.screen_id]
        win_pos = [(thisScreen.width - self.w) / 2,
                   (thisScreen.height - self.h) / 2]
        self.window = pyglet.window.Window(width=self.w, height=self.h, screen=thisScreen, style=win_style)
        # self.window.set_location(int(win_pos[0] + thisScreen.x),
        #                          int(win_pos[1] + thisScreen.y))
        self.window.set_location(int(win_pos[0] + thisScreen.x),
                                 int(win_pos[1] + thisScreen.y))

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
            self._buffer[name + '_arg'] = arg
            self._buffer[name + '_kwargs'] = kwargs
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def get(self, name):
        return self._buffer[name]

    def get_hyper(self, name):
        return self._buffer[name + '_arg'], self._buffer[name + '_kwargs']

    def run(self):
        pyglet.clock.schedule_interval(self.control, 1 / 60.0)
        self.main_App()

    def reg_handlers(self, *args, **kwargs):
        self.handlers_arg = args
        self.handlers_kwargs = kwargs
        self.handlers_exist.clear()

    def control(self, dt):
        if self.pause.is_set():
            self.window.remove_handlers(*self.handlers_arg, **self.handlers_kwargs)
            self.window.clear()
            self.handlers_exist.clear()
            if self.running.is_set():
                self.window.close()
        elif not self.handlers_exist.is_set():
            self.window.push_handlers(*self.handlers_arg, **self.handlers_kwargs)
            self.handlers_exist.set()

    def close_app(self):
        try:
            self.pause.set()
            self.running.set()
        except:
            print("fall to exit")

    def pause_app(self):
        try:
            self.pause.set()
        except:
            print("fall to pause")

    def restart_app(self):
        try:
            self.pause.clear()
            self.handlers_exist.clear()
        except:
            print("fall to restart")
            self.running.clear()

    def pop_window(self, txt, ss):
        root = tk.Tk()
        root.overrideredirect(True)

        frame = tk.Frame(root)

        label = tk.Label(frame, text=txt, font=('Arial', 40), anchor="center")
        label.pack()
        frame.grid()

        x = int((root.winfo_screenwidth() - label.winfo_reqwidth()) / 2)
        y = int((root.winfo_screenheight() - label.winfo_reqheight()) / 2)
        root.geometry("+{}+{}".format(x, y))
        root.after(ss, root.destroy)
        root.mainloop()


class stim_pos_setting(APP):
    def __init__(self, dict, w=1920, h=1080, screen_id=0, *args, **kwargs):

        self.buffer = []
        self.squares = []
        self.square_names = []
        self.editing_square_index = None
        self.current_editing_text = ""  # 当前编辑的文本
        self.key_flag = []
        self.offset_x = 0
        self.offset_y = 0
        self.hide_sq = Event()
        self.hide_sq.clear()
        self.old_hide_flag = self.hide_sq.is_set()

        super().__init__(name='stim_pos_setting', dict=dict, w=w, h=h, screen_id=screen_id, *args, **kwargs)

    def main_App(self):

        self.get_win()
        self.button_setting()
        self.reg_handlers(self.on_draw, self.on_mouse_press, self.on_mouse_drag, self.on_mouse_release, self.on_key_press, self.on_text)
        pyglet.app.run()

    def is_button_clicked(self, button, x, y):
        return button.x < x < button.x + button.width and button.y < y < button.y + button.height

    def button_setting(self):
        # 创建按钮
        self.square_len = 100
        self.button_bigger = pyglet.shapes.Rectangle(x=250, y=100, width=100, height=100)
        self.button_smaller = pyglet.shapes.Rectangle(x=400, y=100, width=100, height=100)
        self.button_add = pyglet.shapes.Rectangle(x=550, y=100, width=100, height=100)
        self.button_del = pyglet.shapes.Rectangle(x=700, y=100, width=100, height=100)
        self.button_save = pyglet.shapes.Rectangle(x=850, y=100, width=100, height=100)
        # self.button_advance = pyglet.shapes.Rectangle(x=1000, y=100, width=100, height=100)
        self.button_newstim = pyglet.shapes.Rectangle(x=1000, y=100, width=100, height=100)
        self.button_quit = pyglet.shapes.Rectangle(x=100, y=self.window.height-100, width=100, height=100)
        self.button_quit_par = pyglet.shapes.Rectangle(x=250, y=self.window.height - 100, width=100, height=100)

        self.button_bigger.color = (127, 127, 127)
        self.button_smaller.color = (127, 127, 127)
        self.button_add.color = (127, 127, 127)
        self.button_del.color = (127, 127, 127)
        self.button_save.color = (127, 127, 127)
        self.button_newstim.color = (127, 127, 0)
        self.button_quit.color = (127, 127, 127)
        self.button_quit_par.color = (127, 127, 127)

        # 创建标签
        self.label_bigger = pyglet.text.Label("Bigger", x=self.button_bigger.x + self.button_bigger.width // 2,
                                              y=self.button_bigger.y + self.button_bigger.height // 2,
                                              anchor_x='center', anchor_y='center')
        self.label_smaller = pyglet.text.Label("Smaller", x=self.button_smaller.x + self.button_smaller.width // 2,
                                               y=self.button_smaller.y + self.button_smaller.height // 2,
                                               anchor_x='center', anchor_y='center')
        self.label_add = pyglet.text.Label("Add", x=self.button_add.x + self.button_add.width // 2,
                                           y=self.button_add.y + self.button_add.height // 2, anchor_x='center',
                                           anchor_y='center')
        self.label_del = pyglet.text.Label("Del", x=self.button_del.x + self.button_del.width // 2,
                                           y=self.button_del.y + self.button_del.height // 2, anchor_x='center',
                                           anchor_y='center')
        self.label_save = pyglet.text.Label("Save", x=self.button_save.x + self.button_save.width // 2,
                                            y=self.button_save.y + self.button_save.height // 2, anchor_x='center',
                                            anchor_y='center')
        self.label_newstim = pyglet.text.Label("New Stim", x=self.button_newstim.x + self.button_newstim.width // 2,
                                               y=self.button_newstim.y + self.button_newstim.height // 2,
                                               anchor_x='center', anchor_y='center')
        self.label_quit= pyglet.text.Label("Close", x=self.button_quit.x + self.button_quit.width // 2,
                                               y=self.button_quit.y + self.button_quit.height // 2,
                                               anchor_x='center', anchor_y='center')
        self.label_quit_par = pyglet.text.Label("Close stim", x=self.button_quit_par.x + self.button_quit_par.width // 2,
                                            y=self.button_quit_par.y + self.button_quit_par.height // 2,
                                            anchor_x='center', anchor_y='center')

    def on_draw(self):
        self.window.clear()
        if not self.hide_sq.is_set():
            for i, square in enumerate(self.squares):
                square.draw()
                label = pyglet.text.Label(
                    self.square_names[i],
                    font_size=200/10,
                    x=square.x + square.width / 2,
                    y=square.y + square.height / 2,
                    anchor_x='center',
                    anchor_y='center'
                )
                if i == self.editing_square_index:
                    label.text = self.current_editing_text
                label.draw()
        self.button_bigger.draw()
        self.button_smaller.draw()
        self.button_add.draw()
        self.button_del.draw()
        self.button_save.draw()
        self.button_newstim.draw()
        self.button_quit.draw()
        self.button_quit_par.draw()
        # 绘制标签
        self.label_bigger.draw()
        self.label_smaller.draw()
        self.label_add.draw()
        self.label_del.draw()
        self.label_save.draw()
        self.label_newstim.draw()
        self.label_quit.draw()
        self.label_quit_par.draw()

    def hide(self):
        self.hide_sq.set()

    def appear(self):
        self.hide_sq.clear()

    def on_mouse_press(self, x, y, input, modifiers):
        if input == mouse.LEFT:
            if self.is_button_clicked(self.button_bigger, x, y):
                # for i, square in enumerate(self.squares):
                #     if self.key_flag[i]:
                #         self.button_bigger.color = (0, 255, 0)  # 设置按钮颜色为绿色
                #         square.width += 10
                #         square.height += 10
                self.square_len += 10
                self.button_bigger.color = (0, 255, 0)
                for square in self.squares:
                    square.width = self.square_len
                    square.height = self.square_len

            elif self.is_button_clicked(self.button_smaller, x, y):
                # for i, square in enumerate(self.squares):
                #     if self.key_flag[i]:
                #         self.button_smaller.color = (0, 255, 0)  # 设置按钮颜色为绿色
                #         square.width -= 10
                #         square.height -= 10
                self.square_len -= 10
                self.button_bigger.color = (0, 255, 0)
                for square in self.squares:
                    square.width = self.square_len
                    square.height = self.square_len

            elif self.is_button_clicked(self.button_add, x, y):
                self.button_add.color = (0, 255, 0)  # 设置按钮颜色为绿色色
                self.squares.append(pyglet.shapes.Rectangle(x=100, y=100, width=self.square_len, height=self.square_len, color=(0, 255, 0)))
                self.square_names.append("Square " + str(len(self.squares)))

                for m in range(len(self.key_flag)):
                    self.key_flag[m] = False
                self.key_flag.append(True)

            elif self.is_button_clicked(self.button_del, x, y):
                self.button_del.color = (0, 255, 0)  # 设置按钮颜色为绿色色
                self.squares = self.squares[:-1]
                self.square_names = self.square_names[:-1]
                self.key_flag = self.key_flag[:-1]
                for m in range(len(self.key_flag)):
                    self.key_flag[m] = False


            elif self.is_button_clicked(self.button_save, x, y):
                '''
                ['paradigm', 'name', 'stim_names', 'stim_pos', 'n_elements',
                'stim_length', 'stim_width','stim_time', 'freqs','phases', 'key_mouse_mapping']
                 or   
                ['paradigm', 'name', 'stim_names', 'rows', 'columns', 'n_elements',
                'stim_length', 'stim_width','stim_time', 'freqs','phases', 'key_mouse_mapping']
                '''
                saver = save_flie()
                self.button_save.color = (0, 255, 0)
                paradigm = 'ssvep'
                stim_names = [name.split(';')[0] for name in self.square_names]
                key_mouse_mapping = [name.split(';')[1] for name in self.square_names]
                stim_pos = []
                n_elements = len(self.square_names)
                stim_length = self.square_len
                stim_width = self.square_len

                #影响性能，暂时不开放自定义
                stim_time = 4.0
                freqs = np.arange(8, 8+n_elements*0.2-0.1, 0.2)  # 目前暂不支持自定义频率
                phases = np.array([(i%2)*0.5 for i in range(n_elements)])

                for i, square in enumerate(self.squares):
                    pos_x = square.x + square.width / 2 - self.win_size[0] / 2
                    pos_y = square.y + square.height / 2 - self.win_size[1] / 2
                    print("Original locations for square ", str(i), " is: x:", square.x, " y:", square.y)
                    stim_pos.append([pos_x, pos_y])

                freqs = freqs.tolist()
                phases = phases.tolist()

                buffer = {'paradigm': paradigm, 'stim_names': stim_names, 'stim_pos': stim_pos, 'n_elements': n_elements,
                        'stim_length': stim_length, 'stim_width': stim_width, 'stim_time': stim_time, 'freqs': freqs,
                          'phases': phases, 'key_mouse_mapping': key_mouse_mapping}
                saver.save(buffer)
                buffer = []

            elif self.is_button_clicked(self.button_newstim, x, y):
                self.button_newstim.color = (0, 255, 0)  # 设置按钮颜色为绿色色
                self.hide()
                self.send('quit_par', True)
                self.send('framework_state', 'showing')
                #self.send('start_par', True)
                #self.send('pause_app', True)

            elif self.is_button_clicked(self.button_quit, x, y):
                self.button_quit.color = (256, 0, 0)  # 设置按钮颜色为绿色色

            elif self.is_button_clicked(self.button_quit_par, x, y):
                self.button_quit_par.color = (256, 0, 0)  # 设置按钮颜色为绿色色

            else:
                for i, square in enumerate(self.squares):
                    if self.is_button_clicked(square, x, y):

                        for m in range(len(self.key_flag)):
                            self.key_flag[m] = False

                        if True not in self.key_flag:
                            self.key_flag[i] = True
                            square.color = (0, 255, 0)
                            self.offset_x = square.x - x
                            self.offset_y = square.y - y
                        else:
                            self.key_flag[i] = False
                    else:
                        self.key_flag[i] = False
                        self.editing_square_index = None

        if input == mouse.RIGHT:
            for i, square in enumerate(self.squares):
                if self.key_flag[i]:
                    self.editing_square_index = i
                    print("edit:",self.editing_square_index)



    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        for i, square in enumerate(self.squares):
            if self.key_flag[i]:
                square.x = x + self.offset_x
                square.y = y + self.offset_y

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            for i, square in enumerate(self.squares):
                if not self.key_flag[i]:
                    square.color = (127, 127, 127)

            if self.is_button_clicked(self.button_bigger, x, y):
                self.button_bigger.color = (127, 127, 127)

            if self.is_button_clicked(self.button_smaller, x, y):
                self.button_smaller.color = (127, 127, 127)

            if self.is_button_clicked(self.button_add, x, y):
                self.button_add.color = (127, 127, 127)

            if self.is_button_clicked(self.button_del, x, y):
                self.button_del.color = (127, 127, 127)

            if self.is_button_clicked(self.button_save, x, y):
                self.button_save.color = (127, 127, 127)

            if self.is_button_clicked(self.button_newstim, x, y):
                self.button_newstim.color = (127, 127, 0)

            if self.is_button_clicked(self.button_quit_par, x, y):
                self.button_quit_par.color = (127, 127, 127)
                self.send('quit_par', True)
                self.send('framework_state', 'started')
                time.sleep(0.1)
                self.appear()


            if self.is_button_clicked(self.button_quit, x, y):
                self.button_quit.color = (127, 127, 127)
                pyglet.app.exit()


    def on_key_press(self, symbol, modifiers):
        if symbol == key.RETURN and self.editing_square_index is not None:
            self.square_names[self.editing_square_index] = self.current_editing_text
            self.current_editing_text = ""
            self.editing_square_index = None
        elif symbol == key.DELETE or symbol == key.BACKSPACE:
            self.current_editing_text = self.current_editing_text[:-1]
    def on_text(self, text):
        if self.editing_square_index is not None:
            self.current_editing_text += text


class menu(APP):
    def __init__(self, dict, w=1920, h=1080, screen_id=0, *args, **kwargs):
        self.window_width = w
        self.window_height = h

        self.selected_line = 0
        self.current_location = "Desk"

        # 创建标签列表
        self.title_list = {}
        self.app_list = {}
        self.device_list = {}
        self.paradigm_list = {}

        super().__init__(name='APP_Menu', dict=dict, w=w, h=h, screen_id=screen_id, *args, **kwargs)

        self.send('goto_menu', None)

    def main_App(self):

        self.get_win(win_style='overlay')

        self.reg_menu_app()

        self.reg_handlers(self.on_draw, self.on_key_press)
        pyglet.app.run()


    def reg_menu_app(self):
        self.title_setting(titles=["General Mode", "System Stims", "Stim KeyBoard Setup", "Device connection", "Stim KeyBoard Management",
                                   "Train Personal Model", "More APP..."],
                           location="Desk")
        self.title_setting(titles=["Glaucoma Detection", "SSVEP Picture Book", "More APP..."],
                           location="Desk/More APP...")

        self.title_setting(titles=["More APP..."],
                           location='Desk/More APP.../More APP...')

        self.title_setting(titles=self.get("device_list"),
                           location='Desk/Device connection')


        self.device_setting(device_list=self.get("device_list"),
                            base_location='Desk/Device connection')

        # self.title_setting(titles=self.get("paradigm_list"),
        #                    location='Desk/General Mode')
        #
        # self.par_setting(par_list=self.get("paradigm_list"),
        #                    base_location='Desk/General Mode')


        self.app_setting(App_name='stim_pos_setting',
                         location='Desk/Stim KeyBoard Setup')
        self.app_setting(App_name='Experiment',
                         location='Desk/Train Personal Model')

    def app_setting(self, App_name: str, location):
        self.app_list[location] = App_name

    def device_setting(self, device_list: list, base_location: str):
        for name in device_list:
            self.device_list[base_location+'/'+name] = name

    def par_setting(self, par_list: list, base_location: str):
        for name in par_list:
            self.paradigm_list[base_location+'/'+name] = name

    def title_setting(self, titles, location):
        labels = []
        line_count = len(titles)
        line_height = self.window_height / (line_count + 4)
        for i in range(line_count):
            label = pyglet.text.Label(
                titles[i],
                font_name='Arial',
                font_size=35,
                x=self.window_width / 6,
                y=self.window_height - (i + 2) * line_height,
                anchor_x='left',
                anchor_y='baseline',
                color=(255, 255, 255, 255)  # 初始颜色为白色
            )
            labels.append(label)
        self.title_list[location] = labels


    def on_draw(self):
        self.window.clear()
        for i, label in enumerate(self.title_list[self.current_location]):
            label.color = (255, 165, 0, 255) \
                if i == self.selected_line else (255, 255, 255, 255)
            label.draw()


    def control(self, dt):
        super().control(dt)
        if self.get('goto_menu') != None:
            self.current_location = self.get('goto_menu')
            self.send('goto_menu', None)

        #更新范式
        paradigm_list_user = [par for par in self.get("paradigm_list") if not par.startswith('#')]
        self.title_setting(titles=paradigm_list_user,
                           location='Desk/General Mode')
        self.par_setting(par_list=paradigm_list_user,
                         base_location='Desk/General Mode')

        paradigm_list_sys = [par for par in self.get("paradigm_list") if par.startswith('#')]
        self.title_setting(titles=paradigm_list_sys,
                           location='Desk/System Stims')
        self.par_setting(par_list=paradigm_list_sys,
                         base_location='Desk/System Stims')


        if self.get("sys_key") == 'up':
            self.selected_line = max(0, self.selected_line - 1)
        elif self.get("sys_key") == 'down':
            self.selected_line = min(len(self.title_list[self.current_location]) - 1, self.selected_line + 1)
        elif self.get("sys_key") == 'enter':
            location = self.current_location + '/' + self.title_list[self.current_location][self.selected_line].text
            if location in self.title_list:
                self.current_location = location
                self.selected_line = 0
            elif location in self.app_list:
                self.send('start_app', self.app_list[location])
                print("start_app:", self.app_list[location])
                self.pause_app()
            elif location in self.device_list:
                self.send("connect_device", self.device_list[location])
                print("connecting_device:", self.device_list[location])
                time.sleep(5)
                if self.get("device_state") == 'connected':
                    self.pop_window('Connected to ' + self.device_list[location], 2000)
                else:
                    self.pop_window('Fail to connect', 2000)
                    self.send("connect_device", None)
            elif location in self.paradigm_list:
                self.send("framework_state", 'hiding')
                if self.get('current_par') != None:
                    self.send('quit_par', True)
                self.send("start_par", self.paradigm_list[location])
                print("start paradigm:", self.paradigm_list[location])
                time.sleep(1)
            else:
                self.pop_window('Coming soon', 2000)

        elif self.get("sys_key") == 'left':
            if len(self.current_location.split('/')) > 1:
                self.current_location = '/'.join(self.current_location.split('/')[:-1])

        self.send('sys_key', None)




    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.selected_line = max(0, self.selected_line - 1)
        elif symbol == pyglet.window.key.DOWN:
            self.selected_line = min(len(self.title_list[self.current_location]) - 1, self.selected_line + 1)
        elif symbol == pyglet.window.key.ENTER:
            location = self.current_location + '/' + self.title_list[self.current_location][self.selected_line].text
            if location in self.title_list:
                self.current_location = location
                self.selected_line = 0
            elif location in self.app_list:
                self.send('start_app', self.app_list[location])
                print("start_app:", self.app_list[location])
                self.pause_app()
            elif location in self.device_list:
                self.send("connect_device", self.device_list[location])
                print("connecting_device:", self.device_list[location])
                time.sleep(5)
                if self.get("device_state") == 'connected':
                    self.pop_window('Connected to ' + self.device_list[location], 2000)
                else:
                    self.pop_window('Fail to connect', 2000)
                    self.send("connect_device", None)
            elif location in self.paradigm_list:
                self.send("framework_state", 'hiding')
                if self.get('current_par') != None:
                    self.send('quit_par', True)
                self.send("start_par", self.paradigm_list[location])
                print("start paradigm:", self.paradigm_list[location])
                time.sleep(1)
            else:
                self.pop_window('Coming soon', 2000)

        elif symbol == pyglet.window.key.LEFT:
            if len(self.current_location.split('/')) > 1:
                self.current_location = '/'.join(self.current_location.split('/')[:-1])


class Experiment(Process):

    DEVICE_COMM_TOOL = {
        'NeuroScan': "NeuroScan",
        'BlueBCI': "Light_trigger",
        'Neuracle': "Neuracle",
    }
    DEVICE_COMM_PORT = {
        'NeuroScan': "COM8",
        'BlueBCI': 1,
        'Neuracle': "COM8",
    }
    DEVICE_SRATE = {
        'BlueBCI': 1000,
    }
    DEVICE_CHANNELS = {
        'BlueBCI': ["PO5", "PO3", "POZ", "PO4", "PO6", "O1", "OZ", "O2"]
    }


    def __init__(self, dict = {} , timeout: float = 5, name: Optional[str] = None,
                 w: int = 1920, h: int = 1080, screen_id: int = 0, *args, **kwargs):
        Process.__init__(self)
        # self.running = Event()
        # self.running.clear()
        # self.pause = Event()
        # self.pause.clear()
        # self.handlers_exist = Event()

        self.timeout = timeout
        self.lock = Lock()
        self.App_name = name
        self._buffer = dict
        self.w = w
        self.h = h
        self.win_size = [self.w, self.h]
        self.screen_id = screen_id
        self.args = args
        self.kwargs = kwargs
        self.parameters = {}
        self.setup_flag = False
        self.experiment_flag = False
        self.next_win = 'mean'


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
            self._buffer[name + '_arg'] = arg
            self._buffer[name + '_kwargs'] = kwargs
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def get(self, name):
        return self._buffer[name]

    def get_hyper(self, name):
        return self._buffer[name + '_arg'], self._buffer[name + '_kwargs']

    def experiment_setting(self):
        def submit():
            paradigm = paradigm_var.get()
            location = location_entry.get()
            if location == 'Default':
                location = None
            experiment_name = experiment_name_entry.get()
            subject = int(subject_entry.get())
            rows = int(rows_entry.get())
            columns = int(columns_entry.get())
            n_elements = int(n_elements_entry.get())
            stim_length = int(stim_length_entry.get())
            stim_width = int(stim_width_entry.get())
            display_time = float(display_time_entry.get())
            rest_time = float(rest_time_entry.get())
            index_time = float(index_time_entry.get())
            response_time = float(response_time_entry.get())
            nrep = int(nrep_entry.get())
            online = online_var.get()
            if online == 'Yes':
                online = True
            else:
                online = False

            print("Values submitted:")
            print(f"Paradigm: {paradigm}")
            print(f"Location: {location}")
            print(f"Experiment name: {experiment_name}")
            print(f"Subject: {subject}")
            print(f"Rows: {rows}")
            print(f"Columns: {columns}")
            print(f"Number of Elements: {n_elements}")
            print(f"Stimulus Length: {stim_length}")
            print(f"Stimulus Width: {stim_width}")
            print(f"Display Time: {display_time}")
            print(f"Rest Time: {rest_time}")
            print(f"Index Time: {index_time}")
            print(f"Response Time: {response_time}")
            print(f"Number of Repetitions: {nrep}")
            print(f"Online: {online}")


            '''
            paradigm, rows and columns, n_elements，stim_length, stim_width,
            display_time, index_time, response_time, nrep, online
            
            由当前设备决定：
            device_type("Light_trigger"...)
            port_addr
            '''

            self.parameters = {
                'paradigm': paradigm,
                'location': location,
                'experiment_name': experiment_name,
                'subject': subject,
                'rows': rows,
                'columns': columns,
                'n_elements': n_elements,
                'stim_length': stim_length,
                'stim_width': stim_width,
                'display_time': display_time,
                'rest_time': rest_time,
                'index_time': index_time,
                'response_time': response_time,
                'nrep': nrep,
                'online': online,
            }


            self.root.destroy()

        # Create a tkinter window
        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.85)
        self.root.title("Experiment Settings")

        # Create labels and entries for each parameter
        parameters = ['Paradigm', 'Location', 'Experiment name', 'Subject', 'Rows', 'Columns', 'Number of Elements', 'Stimulus Length', 'Stimulus Width',
                      'Display Time', 'Rest Time', 'Index Time', 'Response Time', 'Number of Repetitions', 'Online']

        default_values = ['Default', 'Training', '0', '4', '5', '20', '200', '200', '1', '1', '1', '2', '5']

        for i, parameter in enumerate(parameters):
            ttk.Label(self.root, text=parameter).grid(row=i, column=0)

        paradigm_var = tk.StringVar()
        paradigm_combobox = ttk.Combobox(self.root, textvariable=paradigm_var, values=['SSVEP', ])
        paradigm_combobox.current(0)
        paradigm_combobox.grid(row=0, column=1)

        location_entry = ttk.Entry(self.root)
        location_entry.insert(0, default_values[0])  # Insert default value
        location_entry.grid(row=1, column=1)

        experiment_name_entry = ttk.Entry(self.root)
        experiment_name_entry.insert(0, default_values[1])  # Insert default value
        experiment_name_entry.grid(row=2, column=1)

        subject_entry = ttk.Entry(self.root)
        subject_entry.insert(0, default_values[2])  # Insert default value
        subject_entry.grid(row=3, column=1)

        # Insert default values to the Entry widgets
        rows_entry = ttk.Entry(self.root)
        rows_entry.insert(0, default_values[3])  # Insert default value
        rows_entry.grid(row=4, column=1)

        columns_entry = ttk.Entry(self.root)
        columns_entry.insert(0, default_values[4])  # Insert default value
        columns_entry.grid(row=5, column=1)

        n_elements_entry = ttk.Entry(self.root)
        n_elements_entry.insert(0, default_values[5])  # Insert default value
        n_elements_entry.grid(row=6, column=1)

        stim_length_entry = ttk.Entry(self.root)
        stim_length_entry.insert(0, default_values[6])  # Insert default value
        stim_length_entry.grid(row=7, column=1)

        stim_width_entry = ttk.Entry(self.root)
        stim_width_entry.insert(0, default_values[7])  # Insert default value
        stim_width_entry.grid(row=8, column=1)

        display_time_entry = ttk.Entry(self.root)
        display_time_entry.insert(0, default_values[8])  # Insert default value
        display_time_entry.grid(row=9, column=1)

        rest_time_entry = ttk.Entry(self.root)
        rest_time_entry.insert(0, default_values[9])  # Insert default value
        rest_time_entry.grid(row=10, column=1)

        index_time_entry = ttk.Entry(self.root)
        index_time_entry.insert(0, default_values[10])  # Insert default value
        index_time_entry.grid(row=11, column=1)

        response_time_entry = ttk.Entry(self.root)
        response_time_entry.insert(0, default_values[11])  # Insert default value
        response_time_entry.grid(row=12, column=1)

        nrep_entry = ttk.Entry(self.root)
        nrep_entry.insert(0, default_values[12])  # Insert default value
        nrep_entry.grid(row=13, column=1)

        online_var = tk.StringVar()
        online_combobox = ttk.Combobox(self.root, textvariable=online_var, values=['Yes', 'No'])
        online_combobox.current(1)
        online_combobox.grid(row=14, column=1)

        # Submit button
        submit_button = ttk.Button(self.root, text="Submit", command=submit, style='Custom.TButton')
        submit_button.grid(row=15, column=0, columnspan=2, padx=10, pady=10, sticky='we')

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.mainloop()

    def experiment_setup(self):
        self.experiment_setting()
        self.setup_flag = True
        self.next_win = 'mean'


    def training(self, experiment_name, subject):
        from collections import OrderedDict
        import numpy as np
        from scipy.signal import sosfiltfilt
        from sklearn.pipeline import clone
        from sklearn.metrics import balanced_accuracy_score
        from scipy.stats import kurtosis

        from metabci.brainda.datasets import Experiment
        from metabci.brainda.paradigms import SSVEP
        from metabci.brainda.algorithms.utils.model_selection import (
            set_random_seeds,
            generate_loo_indices, match_loo_indices)
        from metabci.brainda.algorithms.decomposition import (
            FBTRCA, FBTDCA, FBSCCA, FBECCA, FBDSP, TRCA, TRCAR,
            generate_filterbank, generate_cca_references)


        dataset = Experiment(experiment_name=experiment_name)
        channels = dataset.channels
        # channels = ['PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'O2']
        srate = dataset.srate  # Hz

        delay = 0  # seconds
        duration = 4  # seconds
        n_bands = 5
        n_harmonics = 4

        events = sorted(list(dataset.events.keys()))
        freqs = [dataset.get_freq(event) for event in events]
        phases = [dataset.get_phase(event) for event in events]

        Yf = generate_cca_references(
            freqs, srate, duration,
            phases=None,
            n_harmonics=n_harmonics)

        start_pnt = dataset.events[events[0]][1][0]
        paradigm = SSVEP(
            srate=srate,
            channels=channels,
            intervals=[(start_pnt + delay, start_pnt + delay + duration + 0.1)],  # more seconds for TDCA
            events=events)

        wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
        ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
        filterbank = generate_filterbank(
            wp, ws, srate, order=4, rp=1)
        filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25

        def data_hook(X, y, meta, caches):
            filterbank = generate_filterbank(
                [[8, 90]], [[6, 95]], srate, order=4, rp=1)
            X = sosfiltfilt(filterbank[0], X, axis=-1)
            return X, y, meta, caches

        paradigm.register_data_hook(data_hook)

        set_random_seeds(64)

        l = 5
        models = OrderedDict([
            ('fbscca', FBSCCA(
                filterbank, filterweights=filterweights)),
            # ('fbecca', FBECCA(
            #     filterbank, filterweights=filterweights)),
            # ('fbdsp', FBDSP(
            #     filterbank, filterweights=filterweights)),
            # ('fbtrca', FBTRCA(
            #     filterbank, filterweights=filterweights)),
            # ('fbtdca', FBTDCA(
            #     filterbank, l, n_components=8,
            #     filterweights=filterweights)),
            # ('trca', TRCA(n_components=1)),
            # ('trcar', TRCAR(n_components=1))
        ])

        X, y, meta = paradigm.get_data(
            dataset,
            subjects=[subject],
            return_concat=True,
            n_jobs=1,
            verbose=False)

        set_random_seeds(42)
        loo_indices = generate_loo_indices(meta)

        ACC_list = {}
        for i, model_name in enumerate(models):
            if model_name == 'fbtdca':
                filterX, filterY = np.copy(X[..., :int(srate * duration) + l]), np.copy(y)
            else:
                filterX, filterY = np.copy(X[..., :int(srate * duration)]), np.copy(y)

            filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

            n_loo = len(loo_indices[subject][events[0]])
            loo_accs = []
            for k in range(n_loo):
                train_ind, validate_ind, test_ind = match_loo_indices(
                    k, meta, loo_indices)
                train_ind = np.concatenate([train_ind, validate_ind])

                trainX, trainY = filterX[train_ind], filterY[train_ind],
                testX, testY = filterX[test_ind], filterY[test_ind]
                # Yf, Yf_Y = filterX[validate_ind], filterY[validate_ind]
                # print("Yf_Y:", Yf_Y)
                model = clone(models[model_name]).fit(
                    trainX, trainY,
                    Yf=Yf
                )
                pred_labels, features = model.predict(testX)
                print("labels:", pred_labels)
                print("kurtosis", kurtosis(np.sort(features), axis=-1, fisher=False))
                loo_accs.append(
                    balanced_accuracy_score(testY, pred_labels))

                self.progress_var.set(int((k + 1) * (1 / n_loo) * 100))
                self.root.update()

            print("Model:{} LOO Acc:{:.2f}".format(model_name, np.mean(loo_accs)))
            ACC_list[model_name] = np.mean(loo_accs)
            self.accuracy_var.set(f"{model_name} Accuracy: {np.mean(loo_accs) * 100}%")
            self.root.update()


        #training for best model
        best_model_name = max(ACC_list)
        if best_model_name == 'fbtdca':
            filterX, filterY = np.copy(X[..., :int(srate * duration) + l]), np.copy(y)
        else:
            filterX, filterY = np.copy(X[..., :int(srate * duration)]), np.copy(y)

        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

        trainX, trainY = filterX[:], filterY[:]

        best_model = clone(models[best_model_name]).fit(
                trainX, trainY,
                Yf=Yf
            )

        return best_model_name, best_model

    def start_experiment(self):
        self.experiment_button.config(state=tk.DISABLED)

        old_workers = self.get("current_workers")
        device = self.get("connect_device")

        self.parameters["device_type"] = self.DEVICE_COMM_TOOL[device]
        self.parameters["port_addr"] = self.DEVICE_COMM_PORT[device]
        self.parameters['channels'] = self.DEVICE_CHANNELS[device]
        self.parameters['srate'] = self.DEVICE_SRATE[device]

        n_elements = self.parameters['n_elements']
        event = [n+1 for n in range(n_elements)]
        self.parameters['freqs'] = np.arange(8, 8 + n_elements * 0.2 - 0.1, 0.2).tolist()  # 目前暂不支持自定义频率
        self.parameters['phases'] = np.array([(i % 4) * 0.5 for i in range(n_elements)]).tolist()


        self.send_hyper("amplifier", device_address=('127.0.0.1', 12345), srate=1000, num_chans=8, use_trigger=True, lsl_source_id="trigger")
        self.send_hyper("marker", interval=[1.4, 5.5], srate=1000, save_data=True, info=self.parameters, clear_after_use=True, events=event,
                        location=self.parameters['location'], experiment_name=self.parameters['experiment_name'], subject=self.parameters['subject'])


        #重新连接设备，进行训练初始化设置
        self.send("connect_device", None)

        while self.get("device_state") == 'connected':
            time.sleep(1)
        print("unconnected, ready to connect to:", device)
        self.send("connect_device", device)

        while self.get("device_state") == 'not_connected':
            time.sleep(1)
        print("connected again, ready to load worker:", device)
        self.send("reg_worker", "EmptyWorker")

        while self.get('reg_worker') != None:
            time.sleep(1)

        #设备和worker重新连接后开始实验
        self.send("framework_type", 'experiment')
        print(self.get("framework_type"))
        self.send('experiment_parameters', self.parameters)
        self.send("quit_par", True)
        self.send("framework_state", 'closed')

        self.root.withdraw()  # 隐藏窗口

        self.send("start_worker", True)

        #等待实验完成后，application framework重启
        while self.get("framework_state") == 'closed':
            time.sleep(1)

        #断开设备为应用模式做准备, 同时在marker也会将实验数据保存
        self.send("connect_device", None)
        while self.get("device_state") == 'connected':
            time.sleep(1)
        print("unconnected, ready to connect to:", device)
        time.sleep(5)

        self.root.deiconify()

        model_name, best_model = self.training(experiment_name=self.parameters['experiment_name'], subject=self.parameters['subject'])


        #model saving 位置暂不支持自定义
        home_dir = os.path.join(os.path.expanduser('~'), 'AssistBCI\\Personal_Model')
        if not os.path.exists(home_dir):
            os.makedirs(home_dir)

        name = model_name + datetime.datetime.now().strftime("_%Y%m%d%H%M%S")
        model_dir = os.path.join(home_dir, name + '.pkl')

        with open(model_dir, 'wb') as file:
            pickle.dump(best_model, file)

        for worker in old_workers:
            self.send("reg_worker", worker)
            while self.get('reg_worker') != None:
                time.sleep(0.1)

        self.send("start_worker", True)

        self.root.destroy()

        sys.exit()

    def goto(self, next_win):
        self.next_win = next_win
        self.root.destroy()

    def main_window_setup(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Training your model")

        # 创建顶部标题
        self.title_label = tk.Label(self.root, text="Training your model", font=("Helvetica", 16))
        self.title_label.pack(pady=10)

        # 创建实验设置部分
        # setup_button = tk.Button(self.main_root, text="Experiment Setup", bg="white", command=lambda: [messagebox.showinfo("Experiment Setup", "Experiment setup completed."), setup_completed()])

        self.setup_button = tk.Button(self.root, text="Experiment Setup", bg="white", command=lambda: self.goto('setup'))
        self.setup_button.pack(pady=5)

        # 创建实验部分
        self.experiment_button = tk.Button(self.root, text="Experiment", command=self.start_experiment, state=tk.DISABLED)
        self.experiment_button.pack(pady=5)

        # 创建训练部分
        self.training_label = tk.Label(self.root, text="Training", font=("Helvetica", 14))
        self.training_label.pack(pady=10)

        # 创建进度条和准确率显示
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.root, length=200, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(pady=5)

        self.accuracy_var = tk.StringVar()
        self.accuracy_label = tk.Label(self.root, textvariable=self.accuracy_var)
        self.accuracy_label.pack(pady=5)

        if self.setup_flag:
            self.setup_button.config(text="Experiment Setup ✓", background="green")
            self.experiment_button.config(state=tk.NORMAL)

        self.root.mainloop()

    def run(self):
        while True:
            if self.next_win == 'mean':
                self.main_window_setup()
            elif self.next_win == 'setup':
                self.experiment_setup()



if __name__ == "__main__":
    a = Experiment()
    a.start()







