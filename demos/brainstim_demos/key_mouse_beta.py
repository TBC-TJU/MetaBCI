from typing import Optional, Any
from abc import abstractmethod
import os
from multiprocessing import Process, Lock ,Event, Queue, Manager
import time
from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as keyboard_controller
from pynput.keyboard import Key
import tkinter as tk

class CMD_Handler(Process):
    DIRECTIONS = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0),
    }

    click_BUTTONS = {  # 第三个命令如：button_left 2 中‘2’代表双击, '1'代表单击，’0‘代表按下或抬起
        'button_left': Button.left,
        'button_right': Button.right,
        'button_middle': Button.middle,
    }

    KEYBOARD = {
        'esc': Key.esc,
        'shift': Key.shift,
        'ctrl': Key.ctrl,
        'alt': Key.alt,
        'alt_l': Key.alt_l,
        'alt_r': Key.alt_r,
        'alt_gr': Key.alt_gr,
        'backspace': Key.backspace,
        'caps_lock': Key.caps_lock,
        'cmd': Key.cmd,
        'cmd_l': Key.cmd_l,
        'cmd_r': Key.cmd_r,
        'ctrl_l': Key.ctrl_l,
        'ctrl_r': Key.ctrl_r,
        'delete': Key.delete,
        'down': Key.down,
        'end': Key.end,
        'enter': Key.enter,
        'f1': Key.f1,
        'f2': Key.f2,
        'f3': Key.f3,
        'f4': Key.f4,
        'f5': Key.f5,
        'f6': Key.f6,
        'f7': Key.f7,
        'f8': Key.f8,
        'f9': Key.f9,
        'f10': Key.f10,
        'f11': Key.f11,
        'f12': Key.f12,
        'f13': Key.f13,
        'f14': Key.f14,
        'f15': Key.f15,
        'f16': Key.f16,
        'f17': Key.f17,
        'f18': Key.f18,
        'f19': Key.f19,
        'f20': Key.f20,
        'home': Key.home,
        'left': Key.left,
        'page_down': Key.page_down,
        'page_up': Key.page_up,
        'right': Key.right,
        'shift_l': Key.shift_l,
        'shift_r': Key.shift_r,
        'space': Key.space,
        'tab': Key.tab,
        'up': Key.up,
        'media_play_pause': Key.media_play_pause,
        'media_volume_mute': Key.media_volume_mute,
        'media_volume_down': Key.media_volume_down,
        'media_volume_up': Key.media_volume_up,
        'media_previous': Key.media_previous,
        'media_next': Key.media_next,
        'insert': Key.insert,
        'menu': Key.menu,
        'num_lock': Key.num_lock,
        'pause': Key.pause,
        'print_screen': Key.print_screen,
        'scroll_lock': Key.scroll_lock
    }

    def __init__(self, dict={}, timeout: float = 5):
        Process.__init__(self)

        self.timeout = timeout
        self.lock = Lock()
        self._buffer = dict
        self._exit = Event()
        self.times = 0
        self.mouse_pressed_flag = False
        self.key_pressed_flag = False
        self.send('CMD_label', None)
        self.send('sys_key', None)
        print(self._buffer)

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




    def mouse_operation(self, command, step='10'):
        """
        Eg:
        command = 'device command 10'
        device -> mouse or keyboard
        command -> what you want to do?
        10 -> operation step
        command including: direction(up/down...), button_(left/right/mid),
        button_(left/right)_(press and release)(note:by set value=0), scroll_(up/down...)
        """


        step = float(step)
        if command in self.DIRECTIONS:
            move_x, move_y = self.DIRECTIONS[command]
            mouse_x, mouse_y = self.mouse.position
            mouse_x += step * move_x
            mouse_y += step * move_y
            self.mouse.position = (mouse_x, mouse_y)
        elif command in self.click_BUTTONS:
            button = self.click_BUTTONS[command]
            if step == 0:
                if not self.mouse_pressed_flag:
                    self.mouse.press(button)
                    self.mouse_pressed_flag = True
                else:
                    self.mouse.release(button)
                    self.mouse_pressed_flag = False
            else:
                self.mouse.click(button, int(step))
        elif command.startswith('scroll_'):
            scroll_direction = command.split('_', 1)[1]
            scroll_vector = tuple(step * x * -1 for x in self.DIRECTIONS[scroll_direction])
            print("scroll:",scroll_vector)
            self.mouse.scroll(scroll_vector[0], scroll_vector[1])
        else:
            return -1  # invalid command

    def keyboard_operation(self, command, aid):
        #如 ctrl+shift+esc  0
        #   ctrl+v          0
        #   a               0.1
        #   b               ~   第一次按下/第二次抬起
        #   123456789       ^   快捷输入

        sep_command = command.split('+')
        try:
            aid = float(aid)
            for cmd in sep_command:
                if cmd in self.KEYBOARD.keys():
                    self.keyboard.press(self.KEYBOARD[cmd])
                else:
                    self.keyboard.press(cmd)
            if aid:
                time.sleep(aid)
            for cmd in reversed(sep_command):
                if cmd in self.KEYBOARD.keys():
                    self.keyboard.release(self.KEYBOARD[cmd])
                else:
                    self.keyboard.release(cmd)
        except:
            if aid == '^':
                self.keyboard.type(command)
            elif aid == '~':
                if not self.key_pressed_flag:
                    for cmd in sep_command:
                        if cmd in self.KEYBOARD.keys():
                            self.keyboard.press(self.KEYBOARD[cmd])
                        else:
                            self.keyboard.press(cmd)
                    self.key_pressed_flag = True
                else:
                    for cmd in reversed(sep_command):
                        if cmd in self.KEYBOARD.keys():
                            self.keyboard.release(self.KEYBOARD[cmd])
                        else:
                            self.keyboard.release(cmd)
                    self.key_pressed_flag = False
            else:
                return -1

    def check(self, command):
        split_command = command.split()
        if len(split_command) != 3:
            print("Invalid CMD")
            # try:
            #     float(split_command[-1])
            #     return True
            # except ValueError:
            return False
        return True

    def stop(self):

        self._exit.set()

    def settimeout(self, timeout=0.01):
        self.timeout = timeout


    def run(self):
        self._exit.clear()
        self.mouse = Controller()
        self.keyboard = keyboard_controller()
        while not self._exit.is_set():
            try:
                self.times += 1
                if self.times == 5:
                    self.times = 0
                    print("listening")

                while self.get('CMD_label') == None:
                    time.sleep(0.1)
                label = self.get("CMD_label")
                self.send("CMD_label", None)
                CMD = self.get('CMD_list')[self.get('current_par')][label]
                if not self.check(CMD):
                    print("Error CMD")
                    continue
                print("receive CMD:", CMD)

                self.pop_window(CMD, 1000)

                CMD = CMD.split()
                if CMD[0] == 'mouse':
                    re = self.mouse_operation(CMD[1], CMD[2])
                    if re == -1:
                        print("invalid mouse CMD")
                elif CMD[0] == 'key':
                    re = self.keyboard_operation(CMD[1], CMD[2])
                    if re == -1:
                        print("invalid key CMD")
                elif CMD[0] == 'sys':
                    try:
                        if CMD[2] == 'True' or CMD[2] == 'False':
                            CMD[2] = eval(CMD[2])
                        self.send(str(CMD[1]), CMD[2])
                    except:
                        print("invalid sys CMD")

            except:
                print(self.get('CMD_label'))
                print("Error in CMD Operation")
                # if queue is empty, loop to wait for next data until exiting
