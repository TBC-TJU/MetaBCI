import math
import time
from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    P300,
    MI,
    AVEP,
    SSAVEP,
    paradigm,
    paradigm_apply,
    pix2height,
    code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix
import pyglet
from pyglet.window import mouse
import ctypes
import multiprocessing
from app_test import stim_pos_setting


# def set_stim_beta(buffer, w=1920, h=1080, screen_id=0, *args, **kwargs):
#
#     # 创建窗口
#     _default_display_ = pyglet.canvas.get_display()
#     allScrs = _default_display_.get_screens()
#     thisScreen = allScrs[screen_id]
#     win_pos = [(thisScreen.width - w) / 2,
#                (thisScreen.height - h) / 2]
#     window = pyglet.window.Window(width=w, height=h, screen=thisScreen, *args, **kwargs)
#     window.set_location(int(win_pos[0] + thisScreen.x),
#                                         int(win_pos[1] + thisScreen.y))
#
#     # # 创建方块
#     # square = pyglet.shapes.Rectangle(x=100, y=100, width=100, height=100)
#
#     squares = []
#     #win_size = np.array([width, height])
#
#     # 创建按钮
#     button_bigger = pyglet.shapes.Rectangle(x=250, y=100, width=100, height=100)
#     button_smaller = pyglet.shapes.Rectangle(x=400, y=100, width=100, height=100)
#     button_add = pyglet.shapes.Rectangle(x=550, y=100, width=100, height=100)
#     button_save = pyglet.shapes.Rectangle(x=700, y=100, width=100, height=100)
#     button_bigger.color = (127, 127, 127)
#     button_smaller.color = (127, 127, 127)
#     button_add.color = (127, 127, 127)
#     button_save.color = (127, 127, 127)
#
#     global key_flag, win_size
#     key_flag = []
#     win_size = [w, h]
#
#     offset_x = 0
#     offset_y = 0
#
#     @window.event
#     def is_button_clicked(button, x, y):
#         return button.x < x < button.x + button.width and button.y < y < button.y + button.height
#
#
#     @window.event
#     def on_draw():
#         window.clear()
#         for square in squares:
#             square.draw()
#         button_bigger.draw()
#         button_smaller.draw()
#         button_add.draw()
#         button_save.draw()
#
#     @window.event
#     def on_mouse_press(x, y, input, modifiers):
#         global key_flag, offset_x, offset_y, buffer, win_size
#         if input == mouse.LEFT:
#             if is_button_clicked(button_bigger, x, y):
#                 for i, square in enumerate(squares):
#                     if key_flag[i]:
#                         button_bigger.color = (0, 255, 0)  # 设置按钮颜色为绿色
#                         square.width += 10
#                         square.height += 10
#
#             elif is_button_clicked(button_smaller, x, y):
#                 for i, square in enumerate(squares):
#                     if key_flag[i]:
#                         button_smaller.color = (0, 255, 0)  # 设置按钮颜色为绿色
#                         square.width -= 10
#                         square.height -= 10
#
#             elif is_button_clicked(button_add, x, y):
#                 button_add.color = (0, 255, 0)  # 设置按钮颜色为绿色色
#                 squares.append(pyglet.shapes.Rectangle(x=100, y=100, width=200, height=200, color=(0, 255, 0)))
#
#                 for m in range(len(key_flag)):
#                     key_flag[m] = False
#                 key_flag.append(True)
#
#
#             elif is_button_clicked(button_save, x, y):
#                 button_save.color = (0, 255, 0)
#                 for i, square in enumerate(squares):
#                     pos_x = square.x + square.width/2 - win_size[0] / 2
#                     pos_y = square.y + square.height/2- win_size[1] / 2
#                     print("Original locations for square ", str(i)," is: x:", square.x, " y:", square.y)
#
#
#                     buffer.append([square.width, square.height, square.color, pos_x, pos_y])
#                 window.close()
#
#
#             else:
#                 for i, square in enumerate(squares):
#                     if is_button_clicked(square, x, y):
#
#                         for m in range(len(key_flag)):
#                             key_flag[m] = False
#
#                         if True not in key_flag:
#                             key_flag[i] = True
#                             square.color = (0, 255, 0)
#                             offset_x = square.x - x
#                             offset_y = square.y - y
#                         else:
#                             key_flag[i] = False
#                     else:
#                         key_flag[i] = False
#
#
#
#     @window.event
#     def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
#         global key_flag, offset_x, offset_y
#         for i, square in enumerate(squares):
#             if key_flag[i]:
#                 square.x = x + offset_x
#                 square.y = y + offset_y
#
#     @window.event
#     def on_mouse_release(x, y, button, modifiers):
#         global key_flag
#         if button == mouse.LEFT:
#             for i, square in enumerate(squares):
#                 if not key_flag[i]:
#                     square.color = (127, 127, 127)
#
#             if is_button_clicked(button_bigger, x, y):
#                 button_bigger.color = (127, 127, 127)
#
#             if is_button_clicked(button_smaller, x, y):
#                 button_smaller.color = (127, 127, 127)
#
#             if is_button_clicked(button_add, x, y):
#                 button_add.color = (127, 127, 127)
#
#     pyglet.app.run()


if __name__ == "__main__":

    ctypes.windll.user32.SetProcessDPIAware()
    win_size = np.array([3120, 2079])
    # set_stim_beta(buffer, win_size[0], win_size[1],
    #               caption="PsychoPy",
    #               fullscreen=False,
    #               screen_id=0,
    #               style='transparent',
    #               )
    #_buffer._init()
    #_buffer.set_value('flag', False)
    dict = multiprocessing.Manager().dict()
    dict['flag'] = False
    app = stim_pos_setting(dict, win_size[0], win_size[1], 0,
                  caption="PsychoPy",
                  fullscreen=False,
                  style='transparent',
                  )
    app.start()

    while not dict['flag']:
        time.sleep(0.5)
    buffer = dict['app1']

    app.stop()

    stim_pos = []
    print("buffer:", buffer)
    n_elements = len(buffer)
    print("n_elements: ", n_elements)
    for data in buffer:
        stim_pos.append(data[-2:])
    print("stim_pos:", stim_pos)
    # stim_pos = [[-250.0, 221.0]]
    # n_elements = 1
    stim_pos = np.array(stim_pos)


    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    mon.setSizePix(win_size)  # 显示器的分辨率
    mon.save()
    bg_color_warm = np.array([0.3, 0.3, 0.3])

    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
        is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
        record_frames=True,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window(win_style='overlay')
    #win = ex.get_window()

    # q退出范式界面
    """
    SSVEP
    """
    # n_elements, rows, columns = 6, 2, 3  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 200, 200  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 90  # 屏幕刷新率
    stim_time = 0.5  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 8+n_elements*1, 1)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        # rows=rows,
        # columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
        stim_pos= stim_pos,
    )
    basic_ssvep.config_text(
        #symbols=["前进","后退","跳跃","下蹲","射击","换弹"],
        tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    #basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 0.1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.2  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  #  0xdefc                                  # 采集主机端口
    port_addr = None  #  0xdefc
    nrep = 200  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSVEP",
        #paradigm,
        paradigm_apply,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        #display_time=display_time,
        #index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        #nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    ex.run()
    app.join()
