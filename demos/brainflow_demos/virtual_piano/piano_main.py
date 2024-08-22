# 导入所需模块
import sys
import os
import pygame
import pygame.midi
from threading import Thread
import time
import mido
import datetime
import chord
import copy
import tkinter as tk
from tkinter import ttk
import configparser
import ctypes
import random
import joblib
from queue import Queue
import numpy as np
import csv
import socket

config = configparser.ConfigParser()
path = 'global_settings.ini'
config.read(path, encoding="UTF-8")

# 全局设置
background_folder_path = config.get('ReadFiles', 'background_folder_path')
font_path = config.get('ReadFiles', 'font_path')
font_settings_path = config.get('ReadFiles', 'font_settings_path')
midi_file_path = config.get('ReadFiles', 'midi_file_path')
light_file_path = config.get('ReadFiles', 'light_file_path')
set_root_from_file = config.getint('SetRootFromFile', 'set_root_from_file')
get_sustain_from_file = config.getint('GetSustainFromFile', 'get_sustain_from_file')
key_light_open = config.getint('KeyLight', 'key_light_open')
light_on_sustain = config.getint('KeyLight', 'light_on_sustain')
light_offset_white_x = config.getint('KeyLight', 'light_offset_white_x')
light_offset_black_x = config.getint('KeyLight', 'light_offset_black_x')
light_offset_y = config.getint('KeyLight', 'light_offset_y')
WKcol = (config.getint('WhiteKeyColor', 'WKcol_R'), config.getint('WhiteKeyColor', 'WKcol_G'),
         config.getint('WhiteKeyColor', 'WKcol_B'))  # 白键按下
BKcol = (config.getint('BlackKeyColor', 'BKcol_R'), config.getint('BlackKeyColor', 'BKcol_G'),
         config.getint('BlackKeyColor', 'BKcol_B'))  # 黑键按下
white_key_waterfall_width = config.getint('WaterFallWidth', 'white_key_waterfall_width')
black_key_waterfall_width = config.getint('WaterFallWidth', 'black_key_waterfall_width')
waterfall_offset_white_key = config.getint('WaterFallOffset', 'waterfall_offset_white_key')
waterfall_offset_black_key = config.getint('WaterFallOffset', 'waterfall_offset_black_key')
NTcol_outline = (config.getint('WaterFallColorMain', 'outline_R'), config.getint('WaterFallColorMain', 'outline_G'),
                 config.getint('WaterFallColorMain', 'outline_B'))  # 瀑布流
NTcol_fill = (config.getint('WaterFallColorMain', 'fill_R'), config.getint('WaterFallColorMain', 'fill_G'),
              config.getint('WaterFallColorMain', 'fill_B'))  # 瀑布流
NTcol2_outline = (config.getint('WaterFallColor2', 'outline_R'), config.getint('WaterFallColor2', 'outline_G'),
                  config.getint('WaterFallColor2', 'outline_B'))  # 瀑布流
NTcol2_fill = (config.getint('WaterFallColor2', 'fill_R'), config.getint('WaterFallColor2', 'fill_G'),
               config.getint('WaterFallColor2', 'fill_B'))  # 瀑布流
NTcol3_outline = (config.getint('WaterFallColor3', 'outline_R'), config.getint('WaterFallColor3', 'outline_G'),
                  config.getint('WaterFallColor3', 'outline_B'))  # 瀑布流
NTcol3_fill = (config.getint('WaterFallColor3', 'fill_R'), config.getint('WaterFallColor3', 'fill_G'),
               config.getint('WaterFallColor3', 'fill_B'))  # 瀑布流
color_boundary_left = config.getint('ColorBoundary', 'color_boundary_left')
color_boundary_middle = config.getint('ColorBoundary', 'color_boundary_middle')
color_boundary_right = config.getint('ColorBoundary', 'color_boundary_right')
black_color_dim_outline = config.getint('BlackColorDim', 'black_color_dim_outline')
black_color_dim_fill = config.getint('BlackColorDim', 'black_color_dim_fill')
WKcol_on_sus = (config.getint('WhiteKeyOnSustain', 'WKcol_sus_R'), config.getint('WhiteKeyOnSustain', 'WKcol_sus_G'),
                config.getint('WhiteKeyOnSustain', 'WKcol_sus_B'))  # 踏板按下，白键松开
BKcol_on_sus = (config.getint('BlackKeyOnSustain', 'BKcol_sus_R'), config.getint('BlackKeyOnSustain', 'BKcol_sus_G'),
                config.getint('BlackKeyOnSustain', 'BKcol_sus_B'))  # 踏板按下，黑键松开
waterfall_color_control = config.getfloat('WaterFallColorControl', 'waterfall_color_control')
time_delta = config.getfloat('TimeDelta', 'time_delta')  # 速度微调（播放MIDI用）
root_delta = config.getfloat('RootDelta', 'root_delta')  # 根音与音符出现的时间差（MODE2专用）
sustain_delta = config.getfloat('SustainDelta', 'sustain_delta')  # 踏板与音符出现的时间差（MODE2专用）
chord_text_color = (
    config.getint('ChordTextColor', 'chord_text_color_R'), config.getint('ChordTextColor', 'chord_text_color_G'),
    config.getint('ChordTextColor', 'chord_text_color_B'))  # 和弦text显示颜色
note_list_text_color = (
    config.getint('NoteListTextColor', 'note_list_text_color_R'),
    config.getint('NoteListTextColor', 'note_list_text_color_G'),
    config.getint('NoteListTextColor', 'note_list_text_color_B'))  # 和弦text显示颜色
sustain_text_color = (
    config.getint('SustainTextColor', 'sustain_text_color_R'),
    config.getint('SustainTextColor', 'sustain_text_color_G'),
    config.getint('SustainTextColor', 'sustain_text_color_B'))  # 踏板text显示颜色
major_key_text_color = (
    config.getint('KeyTextColor', 'major_key_text_color_R'), config.getint('KeyTextColor', 'major_key_text_color_G'),
    config.getint('KeyTextColor', 'major_key_text_color_B'))  # 调性text显示颜色
speed_text_color = (
    config.getint('SpeedTextColor', 'speed_text_color_R'), config.getint('SpeedTextColor', 'speed_text_color_G'),
    config.getint('SpeedTextColor', 'speed_text_color_B'))  # 速度text显示颜色
top_square_color = (
    config.getint('TopSquareColor', 'top_square_color_R'), config.getint('TopSquareColor', 'top_square_color_G'),
    config.getint('TopSquareColor', 'top_square_color_B'))  # 顶端矩形颜色
trans_screen_color = (
    config.getint('TransScreenColor', 'trans_screen_color_R'),
    config.getint('TransScreenColor', 'trans_screen_color_G'),
    config.getint('TransScreenColor', 'trans_screen_color_B'))  # 顶端矩形颜色
transparent_or_not = config.getint('TransparentOrNot', 'transparent_or_not')
trans_screen_opacity = config.getint('TransScreenOpacity', 'trans_screen_opacity')
waterfall_opacity = config.getint('WaterFallOpacity', 'waterfall_opacity')
piano_key_opacity = config.getint('PianoKeyOpacity', 'piano_key_opacity')
top_square_opacity = config.getint('TopSquareOpacity', 'top_square_opacity')
top_square_width = config.getint('TopSquareWidth', 'top_square_width')
global_resolution_x = config.getint('GlobalResolution', 'global_resolution_x')  # 分辨率x（修改可能显示不正常）
global_resolution_y = config.getint('GlobalResolution', 'global_resolution_y')  # 分辨率y
background_offset_x = config.getint('BackGroundOffset', 'bkg_offset_x')
background_offset_y = config.getint('BackGroundOffset', 'bkg_offset_y')
piano_key_offset = config.getint('PianoKeyOffset', 'piano_key_offset')
music_score_offset_x = config.getint('MusicScoreOffset', 'music_score_offset_x')
music_score_offset_y = config.getint('MusicScoreOffset', 'music_score_offset_y')
flash_neon_prepare = config.getint('FlashNeonLight', 'flash_neon_prepare')
flash_neon_pic_path = config.get('FlashNeonLight', 'flash_neon_pic_path')
flash_neon_gap_time = config.getfloat('FlashNeonLight', 'flash_neon_gap_time')

# 字体ini参数
config2 = configparser.ConfigParser()
config2.read(font_settings_path, encoding="UTF-8")
font_size_1 = config2.getint('FontSize', 'font_size_1')
font_size_2 = config2.getint('FontSize', 'font_size_2')
font_size_3 = config2.getint('FontSize', 'font_size_3')
font_size_4 = config2.getint('FontSize', 'font_size_4')
font_size_5 = config2.getint('FontSize', 'font_size_5')
font_size_6 = config2.getint('FontSize', 'font_size_6')
sustain_label_offset_x = config2.getint('SustainLabel', 'sustain_label_offset_x')
sustain_label_offset_y = config2.getint('SustainLabel', 'sustain_label_offset_y')
sustain_state_offset_x = config2.getint('SustainState', 'sustain_state_offset_x')
sustain_state_offset_y = config2.getint('SustainState', 'sustain_state_offset_y')
major_key_offset_x = config2.getint('MajorKey', 'major_key_offset_x')
major_key_offset_y = config2.getint('MajorKey', 'major_key_offset_y')
speed_label_offset_x = config2.getint('SpeedLabel', 'speed_label_offset_x')
speed_label_offset_y = config2.getint('SpeedLabel', 'speed_label_offset_y')
tonicization_offset_x = config2.getint('Tonicization', 'tonicization_offset_x')
tonicization_offset_y = config2.getint('Tonicization', 'tonicization_offset_y')
C_offset = config2.getint('Tonicization', 'C_offset')
Db_offset = config2.getint('Tonicization', 'Db_offset')
D_offset = config2.getint('Tonicization', 'D_offset')
Eb_offset = config2.getint('Tonicization', 'Eb_offset')
E_offset = config2.getint('Tonicization', 'E_offset')
F_offset = config2.getint('Tonicization', 'F_offset')
Gb_offset = config2.getint('Tonicization', 'Gb_offset')
G_offset = config2.getint('Tonicization', 'G_offset')
Ab_offset = config2.getint('Tonicization', 'Ab_offset')
A_offset = config2.getint('Tonicization', 'A_offset')
Bb_offset = config2.getint('Tonicization', 'Bb_offset')
B_offset = config2.getint('Tonicization', 'B_offset')
chord_text_score_offset_y = config2.getint('ChordText', 'chord_text_score_offset_y')
bass_treble_text_offset_y = config2.getint('ChordText', 'bass_treble_text_offset_y')
note_list_text_offset_y = config2.getint('ChordText', 'note_list_text_offset_y')
waterfall_chord_mode_1_x = config2.getint('ChordText', 'waterfall_chord_mode_1_x')
waterfall_chord_mode_1_y = config2.getint('ChordText', 'waterfall_chord_mode_1_y')
waterfall_bass_treble_mode_1_x = config2.getint('ChordText', 'waterfall_bass_treble_mode_1_x')
waterfall_bass_treble_mode_1_y = config2.getint('ChordText', 'waterfall_bass_treble_mode_1_y')
waterfall_note_list_mode_1_x = config2.getint('ChordText', 'waterfall_note_list_mode_1_x')
waterfall_note_list_mode_1_y = config2.getint('ChordText', 'waterfall_note_list_mode_1_y')
waterfall_chord_mode_2_x = config2.getint('ChordText', 'waterfall_chord_mode_2_x')
waterfall_chord_mode_2_y = config2.getint('ChordText', 'waterfall_chord_mode_2_y')
waterfall_bass_treble_mode_2_x = config2.getint('ChordText', 'waterfall_bass_treble_mode_2_x')
waterfall_bass_treble_mode_2_y = config2.getint('ChordText', 'waterfall_bass_treble_mode_2_y')
waterfall_note_list_mode_2_x = config2.getint('ChordText', 'waterfall_note_list_mode_2_x')
waterfall_note_list_mode_2_y = config2.getint('ChordText', 'waterfall_note_list_mode_2_y')
waterfall_chord_mode_3_x = config2.getint('ChordText', 'waterfall_chord_mode_3_x')
waterfall_chord_mode_3_y = config2.getint('ChordText', 'waterfall_chord_mode_3_y')
waterfall_bass_treble_mode_3_x = config2.getint('ChordText', 'waterfall_bass_treble_mode_3_x')
waterfall_bass_treble_mode_3_y = config2.getint('ChordText', 'waterfall_bass_treble_mode_3_y')
waterfall_note_list_mode_3_x = config2.getint('ChordText', 'waterfall_note_list_mode_3_x')
waterfall_note_list_mode_3_y = config2.getint('ChordText', 'waterfall_note_list_mode_3_y')

# 是否从midi文件读取根音
if set_root_from_file == 1:
    root_file_path = config.get('ReadFiles', 'root_file_path')
else:
    root_file_path = ''

# 是否从midi文件读取踏板
if get_sustain_from_file == 1:
    sustain_file_path = config.get('ReadFiles', 'sustain_file_path')
else:
    sustain_file_path = ''

# 从midi文件获取音符数据，播放MIDI用
pygame.init()
pygame.midi.init()

# 获取midi设备数目
midi_count = pygame.midi.get_count()
midi_input_devices = []
midi_input_id = []
midi_output_devices = []
midi_output_id = []
for i in range(midi_count + 1):
    if i == midi_count:
        midi_input_devices.append('None')
        midi_input_id.append(i)
        midi_output_devices.append('None')
        midi_output_id.append(i)
    else:
        if pygame.midi.get_device_info(i)[2] == 1 and pygame.midi.get_device_info(i)[3] == 0:
            midi_input_devices.append(pygame.midi.get_device_info(i)[1].decode('utf-8'))
            midi_input_id.append(i)
        if pygame.midi.get_device_info(i)[2] == 0 and pygame.midi.get_device_info(i)[3] == 1:
            midi_output_devices.append(pygame.midi.get_device_info(i)[1].decode('utf-8'))
            midi_output_id.append(i)

# Tkinter界面
main_window = tk.Tk()
main_window.title("Settings")
main_window.geometry('530x300')
main_window.resizable(0, 0)

# 使用程序自身的dpi适配
ctypes.windll.shcore.SetProcessDpiAwareness(1)

# 创建标签
l1 = tk.Label(main_window, text='Input Device:')
l1.config(font=('Arial', 12))
l1.place(x=8, y=24, height=32, width=150)
l2 = tk.Label(main_window, text='Output Device:')
l2.config(font=('Arial', 12))
l2.place(x=11, y=64, height=32, width=150)
l3 = tk.Label(main_window, text='Play Mode:')
l3.config(font=('Arial', 12))
l3.place(x=0, y=104, height=32, width=150)

# 创建下拉菜单
select_input = ttk.Combobox(main_window, width=80, font=('Arial', 11))
select_output = ttk.Combobox(main_window, width=80, font=('Arial', 11))
select_mode = ttk.Combobox(main_window, width=80, font=('Arial', 11))



# 使用 grid() 来控制控件的位置
select_input.place(x=160, y=24, height=32, width=320)
select_output.place(x=160, y=64, height=32, width=320)
select_mode.place(x=160, y=104, height=32, width=320)

# 设置下拉菜单中的值
if len(midi_input_devices) > 1:
    all_mode = ['MIDI Keyboard + Waterfall', 'MIDI Keyboard + Music Score', 'MIDI Keyboard + Both',
                'MIDI File + Waterfall (rise)', 'MIDI File + Waterfall (fall)', 'MIDI File + Music Score',
                'MIDI File + Both (rise)', 'MIDI File + Both (fall)']
else:
    all_mode = ['MIDI File + Waterfall (rise)', 'MIDI File + Waterfall (fall)', 'MIDI File + Music Score',
                'MIDI File + Both (rise)', 'MIDI File + Both (fall)']
    

select_input['value'] = midi_input_devices
select_output['value'] = midi_output_devices
select_mode['value'] = all_mode

# 设置下拉菜单选项的默认值
select_input.current(0)
select_output.current(0)
select_mode.current(0)

# 获取下拉菜单中选择的值
input_name = select_input.get()
output_name = select_output.get()
mode_name = select_mode.get()

# 获取下拉菜单选择值对应的序号
input_id_pos = 0
output_id_pos = 0
mode_id = 0  # 0-2 MIDI键盘 3-6 MIDI文件
algorithm_id = 0

# 输入选择处理函数
def func1(events):
    global midi_input_devices
    global input_id_pos
    global select_input
    global input_name
    input_name = select_input.get()
    if input_name == 'None':
        input_id_pos = -1
    else:
        for input_id_pos in range(len(midi_input_devices) - 1):
            if midi_input_devices[input_id_pos] == input_name:
                break


# 输出选择处理函数
def func2(events):
    global midi_output_devices
    global output_id_pos
    global select_output
    global output_name
    output_name = select_output.get()
    if output_name == 'None':
        output_id_pos = -1
    else:
        for output_id_pos in range(len(midi_output_devices) - 1):
            if midi_output_devices[output_id_pos] == output_name:
                break


# 模式选择处理函数
def func3(events):
    global all_mode
    global mode_id
    global select_mode
    global mode_name
    mode_name = select_mode.get()
    for mode_id in range(len(all_mode)):
        if all_mode[mode_id] == mode_name:
            break
    # midi input unavailable
    if len(all_mode) == 5:
        mode_id += 3
        # 0-2 MIDI键盘 3-6 MIDI文件



# 绑定下拉菜单事件
select_input.bind("<<ComboboxSelected>>", func1)
select_output.bind("<<ComboboxSelected>>", func2)
select_mode.bind("<<ComboboxSelected>>", func3)

if input_name == 'None':
    input_id_pos = -1
else:
    for input_id_pos in range(len(midi_input_devices) - 1):
        if midi_input_devices[input_id_pos] == input_name:
            break

if output_name == 'None':
    output_id_pos = -1
else:
    for output_id_pos in range(len(midi_output_devices) - 1):
        if midi_output_devices[output_id_pos] == output_name:
            break

for mode_id in range(len(all_mode)):
    if all_mode[mode_id] == mode_name:
        break
 # midi input unavailable
if len(all_mode) == 5:
    mode_id += 3
       



# tkinter窗口主循环
main_window.mainloop()

# midi设备初始化（输入+输出）
if mode_id <= 2:
    if input_id_pos == -1:
        midi1 = 'Unable'
    else:
        midi1 = pygame.midi.Input(midi_input_id[input_id_pos])

if output_id_pos == -1:
    midi2 = 'Unable'
else:
    midi2 = pygame.midi.Output(midi_output_id[output_id_pos])

# 设置主屏窗口
screen = pygame.display.set_mode((global_resolution_x, global_resolution_y))

# 设置窗口标题，窗口icon
pygame.display.set_caption(' Fantasia (Piano Visualizer) ')
img = pygame.image.load("icon/DEFAULT_ICON.ico")
pygame.display.set_icon(img)

pygame.key.stop_text_input()
# 设置背景图片
all_bkg_name = os.listdir(background_folder_path)
bkg_set = 0
bkg_num = len(all_bkg_name)
bkg = pygame.image.load(background_folder_path + '/' + all_bkg_name[bkg_set]).convert()
screen.blit(bkg, (background_offset_x, background_offset_y))

# Neon Light
neon_flash = flash_neon_prepare
all_neon_name = os.listdir('neon/')
neon_light = 0
neon_num = len(all_neon_name)
neon = pygame.image.load('neon/' + all_neon_name[neon_light]).convert_alpha()

# key light
key_light = pygame.image.load(light_file_path).convert_alpha()

# 透明效果
bkg_trans_up = pygame.Surface((global_resolution_x, top_square_width))
bkg_trans_up.set_alpha(top_square_opacity)
bkg_trans_middle = pygame.Surface((global_resolution_x, global_resolution_y - 200 - top_square_width))
bkg_trans_middle.set_alpha(waterfall_opacity)
bkg_trans_down = pygame.Surface((global_resolution_x, 200))
bkg_trans_down.set_alpha(piano_key_opacity)
bkg_trans_up.blit(bkg, (background_offset_x, background_offset_y))
bkg_trans_middle.blit(bkg, (background_offset_x, background_offset_y - top_square_width))
bkg_trans_down.blit(bkg, (background_offset_x, background_offset_y - (global_resolution_y - 200)))

# 音符、五线谱
sheet_w = pygame.image.load('music_score/musicsheets_w.png').convert_alpha()
note_w = pygame.image.load('music_score/note_w.png').convert_alpha()
sharp_w = pygame.image.load('music_score/sharp_w.png').convert_alpha()
flat_w = pygame.image.load('music_score/flat_w.png').convert_alpha()
double_sharp_w = pygame.image.load('music_score/double_sharp_w.png').convert_alpha()
double_flat_w = pygame.image.load('music_score/double_flat_w.png').convert_alpha()
restore_w = pygame.image.load('music_score/restore_w.png').convert_alpha()
line_w = pygame.image.load('music_score/line_w.png').convert_alpha()
line_w_long = pygame.image.load('music_score/line_w_long.png').convert_alpha()

# 半透明
note_w_trans = pygame.image.load('music_score/note_w_trans.png').convert_alpha()
sharp_w_trans = pygame.image.load('music_score/sharp_w_trans.png').convert_alpha()
flat_w_trans = pygame.image.load('music_score/flat_w_trans.png').convert_alpha()
double_sharp_w_trans = pygame.image.load('music_score/double_sharp_w_trans.png').convert_alpha()
double_flat_w_trans = pygame.image.load('music_score/double_flat_w_trans.png').convert_alpha()
restore_w_trans = pygame.image.load('music_score/restore_w_trans.png').convert_alpha()

# 透明遮挡层
trans_screen = pygame.Surface((global_resolution_x, global_resolution_y - 200 - top_square_width))
trans_screen.set_alpha(trans_screen_opacity)
trans_screen.fill(trans_screen_color)

# 相关全局变量
root = -1
check10ms = 0
last_get_10ms = 0
on_sustain = []
sustain = 0
base_time = 0
global_time = 0
global_time_delta = 0
key_note = []  # 0 ~ 87
global_events = []
global_events_lda = []
root_events = []
sustain_events = []
waterfalls = [[] for i in range(88)]
notes_count = [0 for i in range(12)]
appended = [0 for i in range(88)]
all_note_size = 0
major_key = 'Unsettled'
cur_chord = ''
auto_major_key = 1
print_chord = 1
if_exit = 0
waterfall_pos1 = [[0, 0] for i in range(88)]
waterfall_pos2 = [[0, 0] for i in range(88)]
white_key_reflect = []  # 从白键编号到所有键编号
black_key_reflect = []  # 从黑键编号到所有键编号
white_key_pos = [0 for i in range(88)]
black_key_pos1 = [0 for i in range(88)]
black_key_pos2 = [0 for i in range(88)]
white_key_or_not = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
                    1, 0, 1, 1]  # from A
all_note_name = ['A2', 'B2',
                 'C1', 'D1', 'E1', 'F1', 'G1', 'A1', 'B1',
                 'C', 'D', 'E', 'F', 'G', 'A', 'B',
                 'c', 'd', 'e', 'f', 'g', 'a', 'b',
                 'c1', 'd1', 'e1', 'f1', 'g1', 'a1', 'b1',
                 'c2', 'd2', 'e2', 'f2', 'g2', 'a2', 'b2',
                 'c3', 'd3', 'e3', 'f3', 'g3', 'a3', 'b3',
                 'c4', 'd4', 'e4', 'f4', 'g4', 'a4', 'b4',
                 'c5']
note_high_pitch = ['c1', 'd1', 'e1', 'f1', 'g1', 'a1', 'b1',
                   'c2', 'd2', 'e2', 'f2', 'g2', 'a2', 'b2',
                   'c3', 'd3', 'e3', 'f3', 'g3', 'a3', 'b3',
                   'c4', 'd4', 'e4', 'f4', 'g4', 'a4', 'b4',
                   'c5']
note_low_pitch = ['A2', 'B2',
                  'C1', 'D1', 'E1', 'F1', 'G1', 'A1', 'B1',
                  'C', 'D', 'E', 'F', 'G', 'A', 'B',
                  'c', 'd', 'e', 'f', 'g', 'a', 'b']
note_on_sheet = []

# 钢琴键（黑/白键编号与所有键编号对应映射关系）
for i in range(88):
    if white_key_or_not[i] == 1:
        white_key_reflect.append(i)
    else:
        black_key_reflect.append(i)

# 钢琴键位置
for i in range(88):
    if i % 12 == 0:
        # A
        white_key_pos[i] = (37 * (i * 7 / 12)) + piano_key_offset

    if i % 12 == 2:
        # B
        white_key_pos[i] = (37 * (((i - 2) * 7 / 12) + 1)) + piano_key_offset

    if i % 12 == 3:
        # C
        white_key_pos[i] = (37 * (((i - 3) * 7 / 12) + 2)) + piano_key_offset

    if i % 12 == 5:
        # D
        white_key_pos[i] = (37 * (((i - 5) * 7 / 12) + 3)) + piano_key_offset

    if i % 12 == 7:
        # E
        white_key_pos[i] = (37 * (((i - 7) * 7 / 12) + 4)) + piano_key_offset

    if i % 12 == 8:
        # F
        white_key_pos[i] = (37 * (((i - 8) * 7 / 12) + 5)) + piano_key_offset

    if i % 12 == 10:
        # G
        white_key_pos[i] = (37 * (((i - 10) * 7 / 12) + 6)) + piano_key_offset

for i in range(88):
    if i % 12 == 1:
        # Bb
        black_key_pos1[i] = (37 * ((i - 1) * 7 / 12)) + 28 + piano_key_offset
        black_key_pos2[i] = (37 * ((i - 1) * 7 / 12)) + 30 + piano_key_offset

    if i % 12 == 4:
        # Db
        black_key_pos1[i] = (37 * (((i - 4) * 7 / 12) + 2)) + 23 + piano_key_offset
        black_key_pos2[i] = (37 * (((i - 4) * 7 / 12) + 2)) + 25 + piano_key_offset

    if i % 12 == 6:
        # Eb
        black_key_pos1[i] = (37 * (((i - 6) * 7 / 12) + 3)) + 26 + piano_key_offset
        black_key_pos2[i] = (37 * (((i - 6) * 7 / 12) + 3)) + 28 + piano_key_offset

    if i % 12 == 9:
        # Gb
        black_key_pos1[i] = (37 * (((i - 9) * 7 / 12) + 5)) + 22 + piano_key_offset
        black_key_pos2[i] = (37 * (((i - 9) * 7 / 12) + 5)) + 24 + piano_key_offset

    if i % 12 == 11:
        # Ab
        black_key_pos1[i] = (37 * (((i - 11) * 7 / 12) + 6)) + 25 + piano_key_offset
        black_key_pos2[i] = (37 * (((i - 11) * 7 / 12) + 6)) + 27 + piano_key_offset
 
# 瀑布流位置 whitekey 21 blackkey 20
for i in range(88):
    if i % 12 == 0:
        # A
        waterfall_pos1[i][0] = 8 + (37 * (i * 7 / 12)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 9 + (37 * (i * 7 / 12)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 2:
        # B
        waterfall_pos1[i][0] = 11 + (37 * (((i - 2) * 7 / 12) + 1)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 12 + (37 * (((i - 2) * 7 / 12) + 1)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 3:
        # C
        waterfall_pos1[i][0] = 3 + (37 * (((i - 3) * 7 / 12) + 2)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 4 + (37 * (((i - 3) * 7 / 12) + 2)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 5:
        # D
        waterfall_pos1[i][0] = 6 + (37 * (((i - 5) * 7 / 12) + 3)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 7 + (37 * (((i - 5) * 7 / 12) + 3)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 7:
        # E
        waterfall_pos1[i][0] = 9 + (37 * (((i - 7) * 7 / 12) + 4)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 10 + (37 * (((i - 7) * 7 / 12) + 4)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 8:
        # F
        waterfall_pos1[i][0] = 2 + (37 * (((i - 8) * 7 / 12) + 5)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 3 + (37 * (((i - 8) * 7 / 12) + 5)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 10:
        # G
        waterfall_pos1[i][0] = 5 + (37 * (((i - 10) * 7 / 12) + 6)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos2[i][0] = 6 + (37 * (((i - 10) * 7 / 12) + 6)) + piano_key_offset + waterfall_offset_white_key
        waterfall_pos1[i][1] = white_key_waterfall_width
        waterfall_pos2[i][1] = white_key_waterfall_width - 2

    if i % 12 == 1:
        # Bb
        waterfall_pos1[i][0] = 10 + (37 * ((i - 1) * 7 / 12)) + 18 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos2[i][0] = 11 + (37 * ((i - 1) * 7 / 12)) + 18 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos1[i][1] = black_key_waterfall_width
        waterfall_pos2[i][1] = black_key_waterfall_width - 2

    if i % 12 == 4:
        # Db
        waterfall_pos1[i][0] = 9 + (37 * (((i - 4) * 7 / 12) + 2)) + 14 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos2[i][0] = 10 + (37 * (((i - 4) * 7 / 12) + 2)) + 14 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos1[i][1] = black_key_waterfall_width
        waterfall_pos2[i][1] = black_key_waterfall_width - 2

    if i % 12 == 6:
        # Eb
        waterfall_pos1[i][0] = 8 + (37 * (((i - 6) * 7 / 12) + 3)) + 18 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos2[i][0] = 9 + (37 * (((i - 6) * 7 / 12) + 3)) + 18 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos1[i][1] = black_key_waterfall_width
        waterfall_pos2[i][1] = black_key_waterfall_width - 2

    if i % 12 == 9:
        # Gb
        waterfall_pos1[i][0] = 8 + (37 * (((i - 9) * 7 / 12) + 5)) + 14 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos2[i][0] = 9 + (37 * (((i - 9) * 7 / 12) + 5)) + 14 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos1[i][1] = black_key_waterfall_width
        waterfall_pos2[i][1] = black_key_waterfall_width - 2

    if i % 12 == 11:
        # Ab
        waterfall_pos1[i][0] = 9 + (37 * (((i - 11) * 7 / 12) + 6)) + 16 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos2[i][0] = 10 + (37 * (((i - 11) * 7 / 12) + 6)) + 16 + piano_key_offset + waterfall_offset_black_key
        waterfall_pos1[i][1] = black_key_waterfall_width
        waterfall_pos2[i][1] = black_key_waterfall_width - 2

# 各调的Note集合
major_key_note_list = {
    'Unsettled': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'A': [0, 2, 4, 5, 7, 9, 11],
    'Bb': [0, 1, 3, 5, 6, 8, 10],
    'B': [1, 2, 4, 6, 7, 9, 11],
    'C': [0, 2, 3, 5, 7, 8, 10],
    'Db': [1, 3, 4, 6, 8, 9, 11],
    'D': [0, 2, 4, 5, 7, 9, 10],
    'Eb': [1, 3, 5, 6, 8, 10, 11],
    'E': [0, 2, 4, 6, 7, 9, 11],
    'F': [0, 1, 3, 5, 7, 8, 10],
    'Gb': [1, 2, 4, 6, 8, 9, 11],
    'G': [0, 2, 3, 5, 7, 9, 10],
    'Ab': [1, 3, 4, 6, 8, 10, 11]
}

# 音符位置
note_x = 500 + music_score_offset_x
note_x2 = 535 + music_score_offset_x
mod_bas = 455 + music_score_offset_x
mod_dis = 26

# c1~c5
note_high_y = [[428 + music_score_offset_y, True], [416 + music_score_offset_y, False],
               [403 + music_score_offset_y, False], [391 + music_score_offset_y, False],
               [378 + music_score_offset_y, False], [366 + music_score_offset_y, False],
               [353 + music_score_offset_y, False], [341 + music_score_offset_y, False],
               [328 + music_score_offset_y, False], [316 + music_score_offset_y, False],
               [303 + music_score_offset_y, False], [291 + music_score_offset_y, False],
               [278 + music_score_offset_y, True], [266 + music_score_offset_y, False],
               [253 + music_score_offset_y, True], [241 + music_score_offset_y, False],
               [228 + music_score_offset_y, True], [216 + music_score_offset_y, False],
               [203 + music_score_offset_y, True], [191 + music_score_offset_y, False],
               [178 + music_score_offset_y, True], [166 + music_score_offset_y, False],
               [153 + music_score_offset_y, True], [141 + music_score_offset_y, False],
               [128 + music_score_offset_y, True], [116 + music_score_offset_y, False],
               [103 + music_score_offset_y, True], [91 + music_score_offset_y, False],
               [78 + music_score_offset_y, True]]

# A2~b
note_low_y = [[772 + music_score_offset_y, False], [759 + music_score_offset_y, True],
              [747 + music_score_offset_y, False], [734 + music_score_offset_y, True],
              [722 + music_score_offset_y, False], [709 + music_score_offset_y, True],
              [697 + music_score_offset_y, False], [684 + music_score_offset_y, True],
              [672 + music_score_offset_y, False], [659 + music_score_offset_y, True],
              [647 + music_score_offset_y, False], [634 + music_score_offset_y, True],
              [622 + music_score_offset_y, False], [609 + music_score_offset_y, False],
              [597 + music_score_offset_y, False], [584 + music_score_offset_y, False],
              [572 + music_score_offset_y, False], [559 + music_score_offset_y, False],
              [547 + music_score_offset_y, False], [534 + music_score_offset_y, False],
              [522 + music_score_offset_y, False], [509 + music_score_offset_y, False],
              [497 + music_score_offset_y, False]]

# 初始化字体
font1 = pygame.font.Font(font_path, font_size_1)
font2 = pygame.font.Font(font_path, font_size_2)
chord_font = pygame.font.Font(font_path, font_size_3)
note_list_font = pygame.font.Font(font_path, font_size_4)
chord_font_2 = pygame.font.Font(font_path, font_size_5)
note_list_font_2 = pygame.font.Font(font_path, font_size_6)

# 设置字体显示
chord_text = chord_font.render('', True, chord_text_color)
chord_text_2 = chord_font_2.render('', True, chord_text_color)
sustain_label = font1.render('Sustain Pedal', True, sustain_text_color)
major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
speed_print = font1.render('0.0 Notes/Sec', True, speed_text_color)
sustain_state = font2.render('×', True, sustain_text_color)
tonicization_print = font1.render('', True, major_key_text_color)

def get_note():
    global midi_file_path
    global global_events
    start_time = 0
    mid = mido.MidiFile(midi_file_path)
    for p, track in enumerate(mid.tracks):
        for msg0 in track:
            msg = str(msg0)
            print(msg)
            if msg[6] == 'f':
                cur_note = int(msg[msg.find('note=') + 5] + msg[msg.find('note=') + 6] + msg[msg.find('note=') + 7])
                start_time += int(msg[(msg.find('time=') + 5):])
                velocity = 0
                global_events.append([0, cur_note - 21, start_time, velocity])
            elif msg[6] == 'n':
                cur_note = int(msg[msg.find('note=') + 5] + msg[msg.find('note=') + 6] + msg[msg.find('note=') + 7])
                start_time += int(msg[(msg.find('time=') + 5):])
                velocity = 0
                if msg[msg.find('velocity=') + 11] == 't':
                    velocity = int(msg[msg.find('velocity=') + 9] + msg[msg.find('velocity=') + 10])
                elif msg[msg.find('velocity=') + 11] != 't':
                    velocity = int(msg[msg.find('velocity=') + 9] + msg[msg.find('velocity=') + 10] + msg[
                        msg.find('velocity=') + 11])
                global_events.append([1, cur_note - 21, start_time, velocity])
            
# 从指定midi文件读取踏板信号
def get_sustain():
    global sustain_file_path
    global sustain_delta
    global sustain_events
    start_time = 0
    mid = mido.MidiFile(sustain_file_path)
    for p, track in enumerate(mid.tracks):
        for msg0 in track:
            msg = str(msg0)
            if msg[6] == 'f':
                start_time += int(msg[(msg.find('time=') + 5):])
                if mode_id == 3 or mode_id == 5 or mode_id == 6:
                    sustain_events.append([0, start_time])
                elif mode_id == 4 or mode_id == 7:
                    sustain_events.append([0, start_time + root_delta])
            elif msg[6] == 'n':
                start_time += int(msg[(msg.find('time=') + 5):])
                if mode_id == 3 or mode_id == 5 or mode_id == 6:
                    sustain_events.append([1, start_time])
                elif mode_id == 4 or mode_id == 7:
                    sustain_events.append([1, start_time + root_delta])


# 从指定midi文件读取根音
def get_root():
    global root_file_path
    global root_events
    global root_delta
    start_time = 0
    mid = mido.MidiFile(root_file_path)
    for p, track in enumerate(mid.tracks):
        for msg0 in track:
            msg = str(msg0)
            if msg[6] == 'f':
                cur_note = int(msg[msg.find('note=') + 5] + msg[msg.find('note=') + 6] + msg[msg.find('note=') + 7])
                start_time += int(msg[(msg.find('time=') + 5):])
                if mode_id == 3 or mode_id == 5 or mode_id == 6:
                    root_events.append([0, cur_note - 21, start_time])
                elif mode_id == 4 or mode_id == 7:
                    root_events.append([0, cur_note - 21, start_time + root_delta])
            elif msg[6] == 'n':
                cur_note = int(msg[msg.find('note=') + 5] + msg[msg.find('note=') + 6] + msg[msg.find('note=') + 7])
                start_time += int(msg[(msg.find('time=') + 5):])
                if mode_id == 3 or mode_id == 5 or mode_id == 6:
                    root_events.append([1, cur_note - 21, start_time])
                elif mode_id == 4 or mode_id == 7:
                    root_events.append([1, cur_note - 21, start_time + root_delta])


# 调性判断（目前只能判断自然大调，关系小调/Ionian关系中古调式等无法准确显示）
def print_major_key():
    global notes_count
    global major_key
    global major_key_print
    global major_key_text_color
    global auto_major_key
    time_count = 0
    while True:
        if if_exit == 1:
            break
        if auto_major_key == 1:
            max7note = []
            settled = 0
            notes_count_tmp = copy.deepcopy(notes_count)
            for q in range(7):
                mx = -1
                to_select_note = -1
                for r in range(12):
                    if mx < notes_count_tmp[r]:
                        mx = notes_count_tmp[r]
                        to_select_note = r
                if notes_count_tmp[to_select_note] > 0:
                    notes_count_tmp[to_select_note] = 0
                    max7note.append(to_select_note)
            max7note_sorted = sorted(max7note)
            if max7note_sorted == [0, 2, 4, 5, 7, 9, 11]:
                major_key = 'A'
                settled = 1
            elif max7note_sorted == [0, 1, 3, 5, 6, 8, 10]:
                major_key = 'Bb'
                settled = 1
            elif max7note_sorted == [1, 2, 4, 6, 7, 9, 11]:
                major_key = 'B'
                settled = 1
            elif max7note_sorted == [0, 2, 3, 5, 7, 8, 10]:
                major_key = 'C'
                settled = 1
            elif max7note_sorted == [1, 3, 4, 6, 8, 9, 11]:
                major_key = 'Db'
                settled = 1
            elif max7note_sorted == [0, 2, 4, 5, 7, 9, 10]:
                major_key = 'D'
                settled = 1
            elif max7note_sorted == [1, 3, 5, 6, 8, 10, 11]:
                major_key = 'Eb'
                settled = 1
            elif max7note_sorted == [0, 2, 4, 6, 7, 9, 11]:
                major_key = 'E'
                settled = 1
            elif max7note_sorted == [0, 1, 3, 5, 7, 8, 10]:
                major_key = 'F'
                settled = 1
            elif max7note_sorted == [1, 2, 4, 6, 8, 9, 11]:
                major_key = 'Gb'
                settled = 1
            elif max7note_sorted == [0, 2, 3, 5, 7, 9, 10]:
                major_key = 'G'
                settled = 1
            elif max7note_sorted == [1, 3, 4, 6, 8, 10, 11]:
                major_key = 'Ab'
                settled = 1
            if settled == 1:
                notes_count = [0 for m in range(12)]  # 是否需要？
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
        if time_count >= 10:
            time_count = 0
            notes_count = [0 for m in range(12)]
        time_count += 1
        time.sleep(1)


# 获取微秒级时间戳
def get_u_second():
    t = time.time()
    return int(round(t * 1000000))

note_map = {
    0: 64,
    7: 69,
    1: 60,  # C4
    2: 62,  # C#4
    3: 64,  # D4
    4: 65,  # D#4
    5: 67,  # E4 
}

# 从输入设备读取midi信息
import socket
import threading
import time
import queue

# 定义全局变量
cur_midi_signal = queue.Queue()
if_exit = False

def input_midi():
    global if_exit

    # 创建 socket 对象
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 绑定地址和端口
    s.bind(('127.0.0.1', 65432))
    
    # 开始监听
    s.listen()
    print(f"Server listening on {'127.0.0.1'}:{65432}")
    
    try:
        while not if_exit:
            # 接受客户端连接
            client, addr = s.accept()
            print(f"Connected by {addr}")
            
            # 处理客户端发送的数据
            while not if_exit:
                data = client.recv(1024).decode("UTF-8")
                print(f"客户端发来的消息：{data}")
                
                    # 将接收到的数字转换为 MIDI 音符事件
                note = int(data)
                # if note == 0:
                #     pass
                # else:
                velocity = 70  # 力度设置为 70
                    
                    # 将 MIDI 事件添加到队列
                cur_midi_signal.put([144, note_map[note], velocity])
                cur_midi_signal.put([144, note_map[note]-12, velocity])
                cur_midi_signal.put([128, note_map[note], velocity])
                cur_midi_signal.put([128, note_map[note]-12, velocity])
                   
    except KeyboardInterrupt:
        # 捕获中断信号，用于优雅地关闭服务器
        print("Server is shutting down.")
        if_exit = True
    finally:
        # 关闭 socket
        s.close()
        print("Server has stopped.")




# 主循环读取midi信息
def read_midi():
    global cur_midi_signal
    all_midi_data = []
    while not cur_midi_signal.empty():
        all_midi_data.append(cur_midi_signal.get())
    return all_midi_data


# 随机或不随机获取瀑布流颜色
def get_wf_color(key_pos):
    global waterfall_color_control
    to_return = [[0, 0, 0], [0, 0, 0]]
    if waterfall_color_control == 3:
        a = random.randint(150, 215)
        b = random.randint(175, 225)
        c = random.randint(190, 235)
        to_return[0][0] = a
        to_return[0][1] = b
        to_return[0][2] = c
        to_return[1][0] = a + 20
        to_return[1][1] = b + 20
        to_return[1][2] = c + 20
    elif waterfall_color_control == 2:
        # main color
        r_out = NTcol_outline[0]
        g_out = NTcol_outline[1]
        b_out = NTcol_outline[2]
        r_fill = NTcol_fill[0]
        g_fill = NTcol_fill[1]
        b_fill = NTcol_fill[2]
        # color 2
        r2_out = NTcol2_outline[0]
        g2_out = NTcol2_outline[1]
        b2_out = NTcol2_outline[2]
        r2_fill = NTcol2_fill[0]
        g2_fill = NTcol2_fill[1]
        b2_fill = NTcol2_fill[2]
        # color 3
        r3_out = NTcol3_outline[0]
        g3_out = NTcol3_outline[1]
        b3_out = NTcol3_outline[2]
        r3_fill = NTcol3_fill[0]
        g3_fill = NTcol3_fill[1]
        b3_fill = NTcol3_fill[2]
        if key_pos < color_boundary_left:
            to_return[0][0] = r3_out
            to_return[0][1] = g3_out
            to_return[0][2] = b3_out
            to_return[1][0] = r3_fill
            to_return[1][1] = g3_fill
            to_return[1][2] = b3_fill
        elif key_pos > color_boundary_right:
            to_return[0][0] = r_out
            to_return[0][1] = g_out
            to_return[0][2] = b_out
            to_return[1][0] = r_fill
            to_return[1][1] = g_fill
            to_return[1][2] = b_fill
        else:
            to_return[0][0] = r2_out
            to_return[0][1] = g2_out
            to_return[0][2] = b2_out
            to_return[1][0] = r2_fill
            to_return[1][1] = g2_fill
            to_return[1][2] = b2_fill
    elif waterfall_color_control == 1:
        # main color
        r_out = NTcol_outline[0]
        g_out = NTcol_outline[1]
        b_out = NTcol_outline[2]
        r_fill = NTcol_fill[0]
        g_fill = NTcol_fill[1]
        b_fill = NTcol_fill[2]
        # color 3
        r3_out = NTcol3_outline[0]
        g3_out = NTcol3_outline[1]
        b3_out = NTcol3_outline[2]
        r3_fill = NTcol3_fill[0]
        g3_fill = NTcol3_fill[1]
        b3_fill = NTcol3_fill[2]
        if key_pos < color_boundary_middle:
            to_return[0][0] = r3_out
            to_return[0][1] = g3_out
            to_return[0][2] = b3_out
            to_return[1][0] = r3_fill
            to_return[1][1] = g3_fill
            to_return[1][2] = b3_fill
        else:
            to_return[0][0] = r_out
            to_return[0][1] = g_out
            to_return[0][2] = b_out
            to_return[1][0] = r_fill
            to_return[1][1] = g_fill
            to_return[1][2] = b_fill
    else:
        to_return[0] = NTcol_outline
        to_return[1] = NTcol_fill
    if white_key_or_not[key_pos] == 0:
        to_return[0] = [to_return[0][0] - black_color_dim_outline,
                        to_return[0][1] - black_color_dim_outline, to_return[0][2] - black_color_dim_outline]
        to_return[1] = [to_return[1][0] - black_color_dim_fill,
                        to_return[1][1] - black_color_dim_fill, to_return[1][2] - black_color_dim_fill]
    return [tuple(to_return[0]), tuple(to_return[1])]


neon_flashing = neon
def flash_neon():
    global neon_flashing
    neon_flash_dir = os.listdir(flash_neon_pic_path + '/')
    neon_flash_num = len(neon_flash_dir)
    neon_flash_list = []
    neon_flash_dir.sort()
    for i in range(len(neon_flash_dir)):
        neon_flash_list.append(pygame.image.load(flash_neon_pic_path + '/neon (' + str(i + 1) + ').png').convert_alpha())
    while True:
        if if_exit == 1:
            break
        for i in range(len(neon_flash_dir)):
            if if_exit == 1:
                break
            neon_flashing = neon_flash_list[i]
            time.sleep(flash_neon_gap_time)

mode_id = 2
# 若为MIDI播放模式，则读取音符
if mode_id >= 3:
    get_note()
    if set_root_from_file == 1:
        get_root()
    if get_sustain_from_file == 1:
        get_sustain()

# print major key (all mode)
t6 = Thread(target=print_major_key)
t6.start()

# flashing neon
if flash_neon_prepare == 1:
    t7 = Thread(target=flash_neon)
    t7.start()

# if input mode then get midi signals

if mode_id <= 2:
    t8 = Thread(target=input_midi)
    t8.start()

'''
# 测试bug用
def get_note_size():
    while True:
        b = 0
        for i in waterfalls:
            b += len(i)
        print(b)
        time.sleep(0.1)

t9 = Thread(target=get_note_size)
t9.start()
'''

base_time = get_u_second()
time_point = get_u_second()
finished = 0
finished2 = 0
finished3 = 0
cur_pos = 0
cur_pos2 = 0
cur_pos3 = 0

if mode_id == 1 or mode_id == 2 or mode_id == 5 or mode_id == 6 or mode_id == 7:
    print_trans_screen = True
else:
    print_trans_screen = False
timestart = 0
global_time_base = []
# pygame主循环
while True:

    # refresh time
   
    #print("global_time:", global_time_delta)
    # receive midi signals
    
    if mode_id <= 2:
        c_all = read_midi()
        count = 0
        if len(c_all) > 0:
            for midi_event in c_all:
                status, note, velocity = midi_event  # 假设 MIDI 事件是三元组 (状态, 音符, 力度)
                if note == 69:
                    mode_id = 6
                    key_note.clear() 
                    break
                print(note)
                
                
                if status == 144:  # Note on
                    if count % 2 == 0:  # 忽略奇数个 MIDI 事件
                        timestart += 50   #inter
                        global_events_lda.append([1,note,timestart,70])
                        global_events_lda.append([1,note-12,timestart,70])
                    if sustain == 1 and note in on_sustain:
                        on_sustain.remove(note)
                    notes_count[(note % 12)] += 1
                    all_note_size += 1
                 # 模拟 MIDI 按键按下
                    if midi2 != 'Unable':
                        midi2.note_on(note, velocity)
                        time.sleep(0.01)
                        
                    if note not in key_note:  # 避免重复添加
                        key_note.append(note)
                    if mode_id in (0, 2):
                        waterfalls[note].append([0, 0, 0, get_wf_color(note)])
                        
                    #c_all.append([128, note, velocity])  # 假设 MIDI 事件是三元组 (状态, 音符, 力度)
                    
                elif status == 128:  # Note off
                    if count % 2 == 0: 
                        timestart += 50
                        global_events_lda.append([0,note,timestart,70])
                        global_events_lda.append([0,note-12,timestart,70])
                    if sustain == 1:
                        on_sustain.append(note)
                # 模拟 MIDI 按键释放
                    if midi2 != 'Unable':
                        midi2.note_off(note)
                        
                    if note in key_note and len(key_note) > 1:
                        del key_note[:1]
                        if mode_id in (0, 2) and waterfalls[note]:
                            waterfalls[note][-1][0] = 1  # 假设我们更新最后一个元素
                count += 1
    # set note (score only or waterfall up)
    global_time = get_u_second() - base_time
   
    global_time_delta = int(global_time * time_delta / 10000)
    if finished == 0 and (mode_id == 3 or mode_id == 5 or mode_id == 6):
        
        if len(global_time_base)==0:
            global_time_base.append(global_time_delta)
        print(global_events_lda[cur_pos][2]+global_time_base[0], global_time_delta)    
        while global_events_lda[cur_pos][2]+global_time_base[0] <= global_time_delta:
            if global_events_lda[cur_pos][0] == 1:
                if sustain == 1:
                    if global_events_lda[cur_pos][1] in on_sustain:
                        on_sustain.remove(global_events_lda[cur_pos][1])
                        if midi2 != 'Unable':
                            midi2.note_off(global_events_lda[cur_pos][1])
                if midi2 != 'Unable':
                    midi2.note_on(global_events_lda[cur_pos][1], global_events_lda[cur_pos][3])
                notes_count[global_events_lda[cur_pos][1] % 12] += 1
                all_note_size += 1
                key_note.append(global_events_lda[cur_pos][1])
                print(key_note)
                if mode_id == 3 or mode_id == 6:
                    if len(waterfalls[global_events_lda[cur_pos][1]]) > 0:
                        if waterfalls[global_events_lda[cur_pos][1]][-1][0] == 0:
                            waterfalls[global_events_lda[cur_pos][1]][-1][0] = 1
                    waterfalls[global_events_lda[cur_pos][1]].append([0, 0, 0, get_wf_color(global_events_lda[cur_pos][1])])
            elif global_events_lda[cur_pos][0] == 0:
                if sustain == 0:
                    if midi2 != 'Unable':
                        midi2.note_off(global_events_lda[cur_pos][1])
                elif sustain == 1:
                    on_sustain.append(global_events_lda[cur_pos][1])
                if len(key_note) > 0:
                    key_note.remove(global_events_lda[cur_pos][1])
                if mode_id == 3 or mode_id == 6:
                    waterfalls[global_events_lda[cur_pos][1]][len(waterfalls[global_events_lda[cur_pos][1]]) - 1][0] = 1
            cur_pos += 1
            if cur_pos == len(global_events_lda):
                finished = 1
                break
            
    # if finished == 0 and (mode_id == 3 or mode_id == 5 or mode_id == 6):
    #     while global_events[cur_pos][2] <= global_time_delta:
    #         if global_events[cur_pos][0] == 1:
    #             if sustain == 1:
    #                 if global_events[cur_pos][1] + 21 in on_sustain:
    #                     on_sustain.remove(global_events[cur_pos][1] + 21)
    #                     if midi2 != 'Unable':
    #                         midi2.note_off(global_events[cur_pos][1] + 21)
    #             if midi2 != 'Unable':
    #                 midi2.note_on(global_events[cur_pos][1] + 21, global_events[cur_pos][3])
    #             notes_count[global_events[cur_pos][1] % 12] += 1
    #             all_note_size += 1
    #             key_note.append(global_events[cur_pos][1])
    #             print(key_note)
    #             if mode_id == 3 or mode_id == 6:
    #                 if len(waterfalls[global_events[cur_pos][1]]) > 0:
    #                     if waterfalls[global_events[cur_pos][1]][-1][0] == 0:
    #                         waterfalls[global_events[cur_pos][1]][-1][0] = 1
    #                 waterfalls[global_events[cur_pos][1]].append([0, 0, 0, get_wf_color(global_events[cur_pos][1])])
    #         elif global_events[cur_pos][0] == 0:
    #             if sustain == 0:
    #                 if midi2 != 'Unable':
    #                     midi2.note_off(global_events[cur_pos][1] + 21)
    #             elif sustain == 1:
    #                 on_sustain.append(global_events[cur_pos][1] + 21)
    #             if len(key_note) > 0:
    #                 key_note.remove(global_events[cur_pos][1])
    #                 print(key_note)
    #             if mode_id == 3 or mode_id == 6:
    #                 waterfalls[global_events[cur_pos][1]][len(waterfalls[global_events[cur_pos][1]]) - 1][0] = 1
    #         cur_pos += 1
    #         if cur_pos == len(global_events):
    #             finished = 1
    #             break  
        

    # set note (waterfall down)
    if finished == 0 and (mode_id == 4 or mode_id == 7):
        while global_events[cur_pos][2] <= global_time_delta:
            if global_events[cur_pos][0] == 1:
                if len(waterfalls[global_events[cur_pos][1]]) > 0:
                    if waterfalls[global_events[cur_pos][1]][-1][0] == 0:
                        waterfalls[global_events[cur_pos][1]][-1][0] = 1
                waterfalls[global_events[cur_pos][1]].append([0, 0, global_events[cur_pos][3],
                                                              get_wf_color(global_events[cur_pos][1])])
            elif global_events[cur_pos][0] == 0:
                waterfalls[global_events[cur_pos][1]][len(waterfalls[global_events[cur_pos][1]]) - 1][0] = 1
            cur_pos += 1
            if cur_pos == len(global_events):
                finished = 1
                break

    # set root (midi playing)
    if finished2 == 0 and mode_id >= 3 and set_root_from_file == 1:
        while root_events[cur_pos2][2] <= global_time_delta:
            if root_events[cur_pos2][0] == 1:
                root = root_events[cur_pos2][1] % 12
            cur_pos2 += 1
            if cur_pos2 == len(root_events):
                finished2 = 1
                root = -1
                break
                
    # get sustain (midi playing)
    if finished3 == 0 and mode_id >= 3 and get_sustain_from_file == 1:
        while sustain_events[cur_pos3][1] <= global_time_delta:
            if sustain_events[cur_pos3][0] == 1:
                sustain = 1
                sustain_state = font2.render('√', True, sustain_text_color)
            else:
                sustain = 0
                sustain_state = font2.render('×', True, sustain_text_color)
                for y in on_sustain:
                    if (y - 21) not in key_note:
                        if midi2 != 'Unable':
                            midi2.note_off(y)
                on_sustain = []
            cur_pos3 += 1
            if cur_pos3 == len(sustain_events):
                finished3 = 1
                sustain = 0
                break

    # get chord
    cur_chord, note_on_sheet = chord.get_chord(key_note, on_sustain, major_key, root)
    if cur_chord == '':
        chord_text = chord_font.render('(Empty)', True, chord_text_color)
        chord_text_2 = chord_font_2.render('(Empty)', True, chord_text_color)
    else:
        chord_text = chord_font.render(cur_chord, True, chord_text_color)
        chord_text_2 = chord_font_2.render(cur_chord, True, chord_text_color)

    # calculate speed
    delta_t = get_u_second() - time_point
    if delta_t >= 5000000:
        speed_print = font1.render(str(round(all_note_size * 1000000 / delta_t, 1)) + ' Notes/Sec', True,
                                   speed_text_color)
        all_note_size = 0
        time_point = get_u_second()

    # tonicization_check
    if_tonicization = 0
    for i in key_note:
        if (i % 12) not in major_key_note_list[major_key]:
            if_tonicization = 1
            break
    for i in on_sustain:
        if ((i - 21) % 12) not in major_key_note_list[major_key]:
            if_tonicization = 1
            break
    if if_tonicization == 0:
        tonicization_print = font1.render('', True, major_key_text_color)
    elif if_tonicization == 1:
        tonicization_print = font1.render('*', True, major_key_text_color)

    # move waterfall with delete (waterfall up and down)
    if mode_id == 0 or mode_id == 2 or mode_id == 3 or mode_id == 4 or mode_id == 6 or mode_id == 7:
        check10ms = int(global_time / 10000)
        for z in range(check10ms - last_get_10ms):
            last_get_10ms = check10ms
            # ignored time below
            for i in range(len(waterfalls)):
                j = 0
                while True:
                    if j < len(waterfalls[i]):
                        if waterfalls[i][j][0] > 0:
                            waterfalls[i][j][0] += 1
                        waterfalls[i][j][1] += 1
                        # clean old waterfall
                        if waterfalls[i][j][0] > global_resolution_y:
                            waterfalls[i].pop(j)
                    if j >= len(waterfalls[i]):
                        break
                    j += 1

    # print screen
    screen.blit(bkg, (background_offset_x, background_offset_y))

    # print transparent screen (music score mode or decide manually)
    if print_trans_screen:
        screen.blit(trans_screen, (0, top_square_width))

    # print black line above piano keys
    pygame.draw.rect(screen, (100, 100, 100), (0, global_resolution_y - 203, global_resolution_x, 3), 0)

    # print waterfalls (waterfall up)
    if mode_id == 0 or mode_id == 2 or mode_id == 3 or mode_id == 6:
        # white key
        for i in range(88):
            if white_key_or_not[i] == 1:
                for j in waterfalls[i]:
                    pygame.draw.rect(screen, j[3][0], (
                        waterfall_pos1[i][0], global_resolution_y - 200 - j[1], waterfall_pos1[i][1], j[1] - j[0]),
                                     2, border_radius=3)
                    pygame.draw.rect(screen, j[3][1], (waterfall_pos2[i][0], global_resolution_y - 199 - j[1],
                                                       waterfall_pos2[i][1], j[1] - j[0] - 2),
                                     0, border_radius=3)
        # black key
        for i in range(88):
            if white_key_or_not[i] == 0:
                for j in waterfalls[i]:
                    pygame.draw.rect(screen, j[3][0], (
                        waterfall_pos1[i][0], global_resolution_y - 200 - j[1], waterfall_pos1[i][1], j[1] - j[0]),
                                     2, border_radius=3)
                    pygame.draw.rect(screen, j[3][1], (waterfall_pos2[i][0], global_resolution_y - 199 - j[1],
                                                       waterfall_pos2[i][1], j[1] - j[0] - 2),
                                     0, border_radius=3)

    # print waterfalls (waterfall down)
    if mode_id == 4 or mode_id == 7:
        # white key
        for i in range(88):
            if white_key_or_not[i] == 1:
                if len(waterfalls[i]) > 0:
                    if waterfalls[i][0][0] <= (global_resolution_y - 200) <= waterfalls[i][0][1]:
                        if appended[i] == 0:
                            if sustain == 1:
                                if i + 21 in on_sustain:
                                    on_sustain.remove(i + 21)
                                    if midi2 != 'Unable':
                                        midi2.note_off(i + 21)
                            if midi2 != 'Unable':
                                midi2.note_on(i + 21, waterfalls[i][0][2])
                            notes_count[i % 12] += 1
                            all_note_size += 1
                            appended[i] = 1
                            key_note.append(i)
                    elif waterfalls[i][0][0] > (global_resolution_y - 200):
                        if appended[i] == 1:
                            if sustain == 0:
                                if midi2 != 'Unable':
                                    midi2.note_off(i + 21)
                            elif sustain == 1:
                                on_sustain.append(i + 21)
                            if len(key_note) > 0:
                                key_note.remove(i)
                            appended[i] = 0
                        waterfalls[i].pop(0)
                for j in waterfalls[i]:
                    pygame.draw.rect(screen, j[3][0], (waterfall_pos1[i][0], j[0], waterfall_pos1[i][1], j[1] - j[0]),
                                     2, border_radius=3)
                    pygame.draw.rect(screen, j[3][1],
                                     (waterfall_pos2[i][0], j[0] + 1, waterfall_pos2[i][1], j[1] - j[0] - 2), 0,
                                     border_radius=3)
        # black key
        for i in range(88):
            if white_key_or_not[i] == 0:
                if len(waterfalls[i]) > 0:
                    if waterfalls[i][0][0] <= (global_resolution_y - 200) <= waterfalls[i][0][1]:
                        if appended[i] == 0:
                            if sustain == 1:
                                if i + 21 in on_sustain:
                                    on_sustain.remove(i + 21)
                                    if midi2 != 'Unable':
                                        midi2.note_off(i + 21)
                            if midi2 != 'Unable':
                                midi2.note_on(i + 21, waterfalls[i][0][2])
                            notes_count[i % 12] += 1
                            all_note_size += 1
                            appended[i] = 1
                            key_note.append(i)
                    elif waterfalls[i][0][0] > (global_resolution_y - 200):
                        if appended[i] == 1:
                            if sustain == 0:
                                if midi2 != 'Unable':
                                    midi2.note_off(i + 21)
                            elif sustain == 1:
                                on_sustain.append(i + 21)
                            if len(key_note) > 0:
                                key_note.remove(i)
                            appended[i] = 0
                        waterfalls[i].pop(0)
                for j in waterfalls[i]:
                    pygame.draw.rect(screen, j[3][0], (waterfall_pos1[i][0], j[0], waterfall_pos1[i][1], j[1] - j[0]),
                                     2, border_radius=3)
                    pygame.draw.rect(screen, j[3][1],
                                     (waterfall_pos2[i][0], j[0] + 1, waterfall_pos2[i][1], j[1] - j[0] - 2), 0,
                                     border_radius=3)

    # print bottom color
    pygame.draw.rect(screen, (195, 195, 220), (0, global_resolution_y - 200, global_resolution_x, 200), 0)

    # print piano keys
    for i in range(52):
        pygame.draw.rect(screen, 'white', (white_key_pos[white_key_reflect[i]], global_resolution_y - 200, 33, 200), 0)
    for i in key_note:
        if white_key_or_not[i] == 1:
            pygame.draw.rect(screen, WKcol, (white_key_pos[i] + 0, global_resolution_y - 200, 33, 200), 0)
    for h in on_sustain:
        i = h - 21
        if white_key_or_not[i] == 1:
            pygame.draw.rect(screen, WKcol_on_sus, (white_key_pos[i] + 0, global_resolution_y - 200, 33, 200), 0)
    for i in range(36):
        pygame.draw.rect(screen, 'black', (black_key_pos1[black_key_reflect[i]], global_resolution_y - 200, 20, 130), 0)
        pygame.draw.rect(screen, (75, 75, 75),
                         (black_key_pos2[black_key_reflect[i]], global_resolution_y - 200, 15, 120), 0)
    for i in key_note:
        if white_key_or_not[i] == 0:
            pygame.draw.rect(screen, (105, 110, 175), (black_key_pos1[i], global_resolution_y - 200, 20, 130), 0)
            pygame.draw.rect(screen, BKcol, (black_key_pos2[i] + 1, global_resolution_y - 200, 14, 123), 0)
    for h in on_sustain:
        i = h - 21
        if white_key_or_not[i] == 0:
            pygame.draw.rect(screen, (105, 110, 175), (black_key_pos1[i], global_resolution_y - 200, 20, 130), 0)
            pygame.draw.rect(screen, BKcol_on_sus, (black_key_pos2[i] + 1, global_resolution_y - 200, 14, 123), 0)

    # print transparent background
    if transparent_or_not == 1:
        screen.blit(bkg_trans_middle, (0, top_square_width))
        screen.blit(bkg_trans_down, (0, global_resolution_y - 200))

    # print neon light
    if neon_flash == 1:
        screen.blit(neon_flashing, (((global_resolution_x - 1920) / 2), global_resolution_y - 302))
    else:
        if neon_light < neon_num:
            screen.blit(neon, (((global_resolution_x - 1920) / 2), global_resolution_y - 302))

    # print chord (with music score)
    if mode_id == 1 or mode_id == 2 or mode_id == 5 or mode_id == 6 or mode_id == 7:
        screen.blit(chord_text, (1080 + music_score_offset_x, chord_text_score_offset_y + music_score_offset_y))

    # print chord (waterfall single mode)
    if mode_id == 0 or mode_id == 3 or mode_id == 4:
        note_transfer = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        modify_transfer = ['', 'bb', 'b', '#', 'x']
        note_to_print = []
        bass_note = ''
        treble_note = ''
        for h in note_on_sheet:
            if h[2] >= 0:
                to_append = note_transfer[h[0] % 7] + modify_transfer[h[1]]
                if bass_note == '':
                    bass_note = to_append
                treble_note = to_append
                if to_append not in note_to_print:
                    note_to_print.append(to_append)
        if len(treble_note) > 0 and len(bass_note) > 0:
            note_to_print.remove(treble_note)
            if treble_note != bass_note:
                note_to_print.remove(bass_note)
                note_to_print.insert(0, bass_note)
            note_to_print.append(treble_note)
        note_list_text_bt_2 = note_list_font_2.render('Bass: ' + bass_note + '  Treble: ' + treble_note,
                                                  True, note_list_text_color)
        note_list_text_2 = note_list_font_2.render(str(note_to_print), True, note_list_text_color)
        if print_chord == 1:
            screen.blit(chord_text_2, (waterfall_chord_mode_1_x, waterfall_chord_mode_1_y))
            screen.blit(note_list_text_bt_2, (waterfall_bass_treble_mode_1_x, waterfall_bass_treble_mode_1_y))
            screen.blit(note_list_text_2, (waterfall_note_list_mode_1_x, waterfall_note_list_mode_1_y))
        if print_chord == 2:
            screen.blit(chord_text_2, (waterfall_chord_mode_2_x, waterfall_chord_mode_2_y))
            screen.blit(note_list_text_bt_2, (waterfall_bass_treble_mode_2_x, waterfall_bass_treble_mode_2_y))
            screen.blit(note_list_text_2, (waterfall_note_list_mode_2_x, waterfall_note_list_mode_2_y))
        if print_chord == 3:
            screen.blit(chord_text_2, (waterfall_chord_mode_3_x, waterfall_chord_mode_3_y))
            screen.blit(note_list_text_bt_2, (waterfall_bass_treble_mode_3_x, waterfall_bass_treble_mode_3_y))
            screen.blit(note_list_text_2, (waterfall_note_list_mode_3_x, waterfall_note_list_mode_3_y))

    # print notes in list
    if mode_id == 1 or mode_id == 2 or mode_id == 5 or mode_id == 6 or mode_id == 7:
        note_transfer = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        modify_transfer = ['', 'bb', 'b', '#', 'x']
        note_to_print = []
        bass_note = ''
        treble_note = ''
        for h in note_on_sheet:
            if h[2] >= 0:
                to_append = note_transfer[h[0] % 7] + modify_transfer[h[1]]
                if bass_note == '':
                    bass_note = to_append
                treble_note = to_append
                if to_append not in note_to_print:
                    note_to_print.append(to_append)
        if len(treble_note) > 0 and len(bass_note) > 0:
            note_to_print.remove(treble_note)
            if treble_note != bass_note:
                note_to_print.remove(bass_note)
                note_to_print.insert(0, bass_note)
            note_to_print.append(treble_note)
        note_list_text_bt = note_list_font.render('Bass: ' + bass_note + '  Treble: ' + treble_note,
                                                  True, note_list_text_color)
        note_list_text = note_list_font.render(str(note_to_print), True, note_list_text_color)
        screen.blit(note_list_text_bt, (1080 + music_score_offset_x, bass_treble_text_offset_y + music_score_offset_y))
        screen.blit(note_list_text, (1080 + music_score_offset_x, note_list_text_offset_y + music_score_offset_y))

    # print musicsheets and notes (music score)
    if mode_id == 1 or mode_id == 2 or mode_id == 5 or mode_id == 6 or mode_id == 7:
        screen.blit(sheet_w, (-210 + music_score_offset_x, -230 + music_score_offset_y))
        highest_note = -1
        lowest_note = -1
        if len(note_on_sheet) > 0:
            highest_note = note_on_sheet[len(note_on_sheet) - 1][0]
            lowest_note = note_on_sheet[0][0]
        cur_x_low = note_x
        cur_x_high = note_x
        cur_x_low_last = False
        cur_x_high_last = False
        draw_line_low = [lowest_note, 0, 0]
        draw_line_high = [highest_note, 0, 0]
        draw_line_mid_c = [0, 0]
        last_k_low = -1
        last_k_high = -1
        have_tag_pos = []
        mod_x_low = mod_bas
        mod_x_high = mod_bas
        count_space_low = 0
        count_space_high = 0
        have_flat_low = []
        have_sharp_low = []
        have_flat_high = []
        have_sharp_high = []
        tonality_dict = {'Unsettled': 0, 'C': 0, 'F': 1, 'Bb': 2, 'Eb': 3, 'Ab': 4, 'Db': 5, 'Gb': 6,
                         'G': 7, 'D': 8, 'A': 9, 'E': 10, 'B': 11}
        flat_note_low = [15, 18, 14, 17, 13, 16]
        sharp_note_low = [19, 16, 20, 17, 14]
        flat_note_high = [6, 9, 5, 8, 4, 7]
        sharp_note_high = [10, 7, 11, 8, 5]

        # print sharp and flat tags based on tonality
        print_time = tonality_dict[major_key]
        if print_time <= 6:
            for s in range(print_time):
                screen.blit(flat_w, (mod_x_low - 190 + s * 18, note_low_y[flat_note_low[s]][0]))
                have_flat_low.append(flat_note_low[s] % 7)
                screen.blit(flat_w, (mod_x_high - 190 + s * 18, note_high_y[flat_note_high[s]][0]))
                have_flat_high.append(flat_note_high[s] % 7)
        else:
            for s in range(print_time - 6):
                screen.blit(sharp_w, (mod_x_low - 190 + s * 18, note_low_y[sharp_note_low[s]][0]))
                have_sharp_low.append(sharp_note_low[s] % 7)
                screen.blit(sharp_w, (mod_x_high - 190 + s * 18, note_high_y[sharp_note_high[s]][0]))
                have_sharp_high.append(sharp_note_high[s] % 7)

        # print note
        for i in range(len(note_on_sheet)):
            if i >= len(note_on_sheet):
                break
            k = note_on_sheet[i]
            if k[2] == -1:
                pass
            elif k[0] < 28:
                # 低音区
                if k[0] - last_k_low <= 1:
                    if cur_x_low == note_x2:
                        cur_x_low = note_x
                    else:
                        cur_x_low = note_x2
                else:
                    cur_x_low = note_x
                screen.blit((note_w if k[2] == 1 else note_w_trans), (cur_x_low, note_low_y[k[0] - 5][0]))
                if (note_low_y[k[0] - 5][0]) >= 634 + music_score_offset_y:
                    if cur_x_low == note_x:
                        draw_line_low[1] = 1
                    elif cur_x_low == note_x2:
                        draw_line_low[2] = 1
                last_k_low = k[0]
                # 低音区变音记号
                if k[1] == 1:
                    mod_x_low = mod_bas
                    while True:
                        tag_have_space = True
                        for pos in have_tag_pos:
                            if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                tag_have_space = False
                                break
                        if tag_have_space:
                            break
                        else:
                            mod_x_low -= mod_dis
                    screen.blit((double_flat_w if k[2] == 1 else double_flat_w_trans),
                                (mod_x_low, note_low_y[k[0] - 5][0]))
                    have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])
                elif k[1] == 2:
                    if (k[0] - 5) % 7 not in have_flat_low:
                        mod_x_low = mod_bas
                        while True:
                            tag_have_space = True
                            for pos in have_tag_pos:
                                if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                    tag_have_space = False
                                    break
                            if tag_have_space:
                                break
                            else:
                                mod_x_low -= mod_dis
                        screen.blit((flat_w if k[2] == 1 else flat_w_trans), (mod_x_low, note_low_y[k[0] - 5][0]))
                        have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])
                elif k[1] == 3:
                    if (k[0] - 5) % 7 not in have_sharp_low:
                        mod_x_low = mod_bas
                        while True:
                            tag_have_space = True
                            for pos in have_tag_pos:
                                if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                    tag_have_space = False
                                    break
                            if tag_have_space:
                                break
                            else:
                                mod_x_low -= mod_dis
                        screen.blit((sharp_w if k[2] == 1 else sharp_w_trans), (mod_x_low, note_low_y[k[0] - 5][0]))
                        have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])
                elif k[1] == 4:
                    mod_x_low = mod_bas
                    while True:
                        tag_have_space = True
                        for pos in have_tag_pos:
                            if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                tag_have_space = False
                                break
                        if tag_have_space:
                            break
                        else:
                            mod_x_low -= mod_dis
                    screen.blit((double_sharp_w if k[2] == 1 else double_sharp_w_trans),
                                (mod_x_low, note_low_y[k[0] - 5][0]))
                    have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])
                elif k[1] == 0:
                    if (k[0] - 5) % 7 not in have_sharp_low and (k[0] - 5) % 7 not in have_flat_low:
                        if i - 1 >= 0 and len(note_on_sheet) > 0:
                            if k[0] == note_on_sheet[i - 1][0] and note_on_sheet[i - 1][2] != -1:
                                mod_x_low = mod_bas
                                while True:
                                    tag_have_space = True
                                    for pos in have_tag_pos:
                                        if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                            tag_have_space = False
                                            break
                                    if tag_have_space:
                                        break
                                    else:
                                        mod_x_low -= mod_dis
                                screen.blit((restore_w if k[2] == 1 else restore_w_trans),
                                            (mod_x_low, note_low_y[k[0] - 5][0]))
                                have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])
                        if i + 1 < len(note_on_sheet):
                            if note_on_sheet[i + 1][0] == k[0] and note_on_sheet[i + 1][2] != -1:
                                mod_x_low = mod_bas
                                while True:
                                    tag_have_space = True
                                    for pos in have_tag_pos:
                                        if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                            tag_have_space = False
                                            break
                                    if tag_have_space:
                                        break
                                    else:
                                        mod_x_low -= mod_dis
                                screen.blit((restore_w if k[2] == 1 else restore_w_trans),
                                            (mod_x_low, note_low_y[k[0] - 5][0]))
                                have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])
                    else:
                        mod_x_low = mod_bas
                        while True:
                            tag_have_space = True
                            for pos in have_tag_pos:
                                if pos[0] == mod_x_low and abs(pos[1] - note_low_y[k[0] - 5][0]) <= 25:
                                    tag_have_space = False
                                    break
                            if tag_have_space:
                                break
                            else:
                                mod_x_low -= mod_dis
                        screen.blit((restore_w if k[2] == 1 else restore_w_trans), (mod_x_low, note_low_y[k[0] - 5][0]))
                        have_tag_pos.append([mod_x_low, note_low_y[k[0] - 5][0]])

            else:
                # 高音区
                if k[0] - last_k_high <= 1:
                    if cur_x_high == note_x2:
                        cur_x_high = note_x
                    else:
                        cur_x_high = note_x2
                else:
                    cur_x_high = note_x
                screen.blit((note_w if k[2] == 1 else note_w_trans), (cur_x_high, note_high_y[k[0] - 28][0]))
                if note_high_y[k[0] - 28][0] <= 278 + music_score_offset_y:
                    if cur_x_high == note_x:
                        draw_line_high[1] = 1
                    elif cur_x_high == note_x2:
                        draw_line_high[2] = 1
                elif note_high_y[k[0] - 28][0] == 428 + music_score_offset_y:
                    if cur_x_high == note_x:
                        draw_line_mid_c[0] = 1
                    elif cur_x_high == note_x2:
                        draw_line_mid_c[1] = 1
                last_k_high = k[0]
                # 高音区变音记号
                if k[1] == 1:
                    mod_x_high = mod_bas
                    while True:
                        tag_have_space = True
                        for pos in have_tag_pos:
                            if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                tag_have_space = False
                                break
                        if tag_have_space:
                            break
                        else:
                            mod_x_high -= mod_dis
                    screen.blit((double_flat_w if k[2] == 1 else double_flat_w_trans),
                                (mod_x_high, note_high_y[k[0] - 28][0]))
                    have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])
                elif k[1] == 2:
                    if (k[0] - 28) % 7 not in have_flat_high:
                        mod_x_high = mod_bas
                        while True:
                            tag_have_space = True
                            for pos in have_tag_pos:
                                if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                    tag_have_space = False
                                    break
                            if tag_have_space:
                                break
                            else:
                                mod_x_high -= mod_dis
                        screen.blit((flat_w if k[2] == 1 else flat_w_trans), (mod_x_high, note_high_y[k[0] - 28][0]))
                        have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])
                elif k[1] == 3:
                    if (k[0] - 28) % 7 not in have_sharp_high:
                        mod_x_high = mod_bas
                        while True:
                            tag_have_space = True
                            for pos in have_tag_pos:
                                if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                    tag_have_space = False
                                    break
                            if tag_have_space:
                                break
                            else:
                                mod_x_high -= mod_dis
                        screen.blit((sharp_w if k[2] == 1 else sharp_w_trans), (mod_x_high, note_high_y[k[0] - 28][0]))
                        have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])
                elif k[1] == 4:
                    mod_x_high = mod_bas
                    while True:
                        tag_have_space = True
                        for pos in have_tag_pos:
                            if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                tag_have_space = False
                                break
                        if tag_have_space:
                            break
                        else:
                            mod_x_high -= mod_dis
                    screen.blit((double_sharp_w if k[2] == 1 else double_sharp_w_trans),
                                (mod_x_high, note_high_y[k[0] - 28][0]))
                    have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])
                elif k[1] == 0:
                    if (k[0] - 28) % 7 not in have_sharp_high and (k[0] - 28) % 7 not in have_flat_high:
                        if i - 1 >= 0 and len(note_on_sheet) > 0:
                            if k[0] == note_on_sheet[i - 1][0]:
                                mod_x_high = mod_bas
                                while True:
                                    tag_have_space = True
                                    for pos in have_tag_pos:
                                        if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                            tag_have_space = False
                                            break
                                    if tag_have_space:
                                        break
                                    else:
                                        mod_x_high -= mod_dis
                                screen.blit((restore_w if k[2] == 1 else restore_w_trans),
                                            (mod_x_high, note_high_y[k[0] - 28][0]))
                                have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])
                        if i + 1 < len(note_on_sheet):
                            if note_on_sheet[i + 1][0] == k[0]:
                                mod_x_high = mod_bas
                                while True:
                                    tag_have_space = True
                                    for pos in have_tag_pos:
                                        if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                            tag_have_space = False
                                            break
                                    if tag_have_space:
                                        break
                                    else:
                                        mod_x_high -= mod_dis
                                screen.blit((restore_w if k[2] == 1 else restore_w_trans),
                                            (mod_x_high, note_high_y[k[0] - 28][0]))
                                have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])
                    else:
                        mod_x_high = mod_bas
                        while True:
                            tag_have_space = True
                            for pos in have_tag_pos:
                                if pos[0] == mod_x_high and abs(pos[1] - note_high_y[k[0] - 28][0]) <= 25:
                                    tag_have_space = False
                                    break
                            if tag_have_space:
                                break
                            else:
                                mod_x_high -= mod_dis
                        screen.blit((restore_w if k[2] == 1 else restore_w_trans),
                                    (mod_x_high, note_high_y[k[0] - 28][0]))
                        have_tag_pos.append([mod_x_high, note_high_y[k[0] - 28][0]])

        # print line
        start_draw_low = False
        stop_draw_high = False
        for k in range(5, 57):
            if k < 28:
                # 低音区
                if note_low_y[k - 5][0] >= 634 + music_score_offset_y:
                    if k == draw_line_low[0]:
                        start_draw_low = True
                    if note_low_y[k - 5][1] and start_draw_low:
                        if draw_line_low[1] == 1 and draw_line_low[2] == 0:
                            screen.blit(line_w, (note_x, note_low_y[k - 5][0]))
                        elif draw_line_low[2] == 1:
                            screen.blit(line_w_long, (note_x, note_low_y[k - 5][0]))
            else:
                # 高音区
                if note_high_y[k - 28][0] == 428 + music_score_offset_y:
                    if draw_line_mid_c[0] == 1:
                        screen.blit(line_w, (note_x, note_high_y[k - 28][0]))
                    if draw_line_mid_c[1] == 1:
                        screen.blit(line_w, (note_x2, note_high_y[k - 28][0]))
                if note_high_y[k - 28][0] <= 278 + music_score_offset_y:
                    if note_high_y[k - 28][1] and not stop_draw_high:
                        if draw_line_high[1] == 1 and draw_line_high[2] == 0:
                            screen.blit(line_w, (note_x, note_high_y[k - 28][0]))
                        elif draw_line_high[2] == 1:
                            screen.blit(line_w_long, (note_x, note_high_y[k - 28][0]))
                    if k == draw_line_high[0]:
                        stop_draw_high = True

    # print square
    font_distance = {
        'Unsettled': 99999,
        'C': C_offset,
        'Db': Db_offset,
        'D': D_offset,
        'Eb': Eb_offset,
        'E': E_offset,
        'F': F_offset,
        'Gb': Gb_offset,
        'G': G_offset,
        'Ab': Ab_offset,
        'A': A_offset,
        'Bb': Bb_offset,
        'B': B_offset
    }

    # 顶端矩形
    pygame.draw.rect(screen, top_square_color, (0, 0, global_resolution_x, top_square_width), 0)

    # 顶端矩形透明度调整（背景覆盖）
    if transparent_or_not == 1:
        screen.blit(bkg_trans_up, (0, 0))

    # print key light
    if key_light_open == 1:
        for i in key_note:
            if white_key_or_not[i] == 1:
                screen.blit(key_light, (white_key_pos[i] + light_offset_white_x, global_resolution_y + light_offset_y))
        for i in key_note:
            if white_key_or_not[i] == 0:
                screen.blit(key_light, (black_key_pos1[i] + light_offset_black_x, global_resolution_y + light_offset_y))
        if light_on_sustain == 1:
            for s in on_sustain:
                i = s - 21
                if white_key_or_not[i] == 1:
                    screen.blit(key_light, (white_key_pos[i] + light_offset_white_x, global_resolution_y + light_offset_y))
            for s in on_sustain:
                i = s - 21
                if white_key_or_not[i] == 0:
                    screen.blit(key_light, (black_key_pos1[i] + light_offset_black_x, global_resolution_y + light_offset_y))

    # 显示文字内容
    screen.blit(sustain_label, (global_resolution_x - sustain_label_offset_x, sustain_label_offset_y))
    screen.blit(sustain_state, (global_resolution_x - sustain_state_offset_x, sustain_state_offset_y))
    screen.blit(major_key_print, (major_key_offset_x, major_key_offset_y))
    screen.blit(speed_print, ((global_resolution_x / 2) - speed_label_offset_x, speed_label_offset_y))
    screen.blit(tonicization_print, (tonicization_offset_x + font_distance[major_key], tonicization_offset_y))

    # 循环获取事件，监听事件状态
    for event in pygame.event.get():
        # 判断用户是否点了"X"关闭按钮,并执行if代码段
        if event.type == pygame.QUIT:
            if_exit = 1
            # 卸载所有模块
            pygame.quit()
            # 终止程序，确保退出程序
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                major_key = 'Unsettled'
                auto_major_key = 1
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_m:
                if mode_id <= 2:
                    mode_id += 1
                    if mode_id > 2:
                        mode_id -= 3
                    if mode_id == 0:
                        print_trans_screen = False
                    else:
                        print_trans_screen = True
                    waterfalls = [[] for i in range(88)]
            elif event.key == pygame.K_c:
                if major_key == 'Unsettled':
                    major_key = 'C'
                elif major_key == 'C':
                    major_key = 'Db'
                elif major_key == 'Db':
                    major_key = 'D'
                elif major_key == 'D':
                    major_key = 'Eb'
                elif major_key == 'Eb':
                    major_key = 'E'
                elif major_key == 'E':
                    major_key = 'F'
                elif major_key == 'F':
                    major_key = 'Gb'
                elif major_key == 'Gb':
                    major_key = 'G'
                elif major_key == 'G':
                    major_key = 'Ab'
                elif major_key == 'Ab':
                    major_key = 'A'
                elif major_key == 'A':
                    major_key = 'Bb'
                elif major_key == 'Bb':
                    major_key = 'B'
                elif major_key == 'B':
                    major_key = 'C'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_b:
                if major_key == 'Unsettled':
                    major_key = 'C'
                elif major_key == 'C':
                    major_key = 'B'
                elif major_key == 'B':
                    major_key = 'Bb'
                elif major_key == 'Bb':
                    major_key = 'A'
                elif major_key == 'A':
                    major_key = 'Ab'
                elif major_key == 'Ab':
                    major_key = 'G'
                elif major_key == 'G':
                    major_key = 'Gb'
                elif major_key == 'Gb':
                    major_key = 'F'
                elif major_key == 'F':
                    major_key = 'E'
                elif major_key == 'E':
                    major_key = 'Eb'
                elif major_key == 'Eb':
                    major_key = 'D'
                elif major_key == 'D':
                    major_key = 'Db'
                elif major_key == 'Db':
                    major_key = 'C'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_u:
                major_key = 'Unsettled'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_p:
                print_chord += 1
                if print_chord > 3:
                    print_chord = 0
            elif event.key == pygame.K_d:
                auto_major_key = 0
            elif event.key == pygame.K_s:
                print_trans_screen = not print_trans_screen
            elif event.key == pygame.K_t:
                transparent_or_not = 1 - transparent_or_not
            elif event.key == pygame.K_r:
                waterfall_color_control += 1
                if waterfall_color_control >= 4:
                    waterfall_color_control -= 4
            elif event.key == pygame.K_g:
                bkg_set += 1
                if bkg_set >= bkg_num:
                    bkg_set -= bkg_num
                bkg = pygame.image.load(background_folder_path + '/' + all_bkg_name[bkg_set]).convert()
                bkg_trans_up.blit(bkg, (background_offset_x, background_offset_y))
                bkg_trans_middle.blit(bkg, (background_offset_x, background_offset_y - top_square_width))
                bkg_trans_down.blit(bkg, (background_offset_x, background_offset_y - (global_resolution_y - 200)))
            elif event.key == pygame.K_h:
                key_light_open = 1 - key_light_open
            elif event.key == pygame.K_l:
                neon_light += 1
                if neon_light >= neon_num + 1:
                    neon_light -= (neon_num + 1)
                if neon_light < neon_num:
                    neon = pygame.image.load('neon/' + all_neon_name[neon_light]).convert_alpha()
            elif event.key == pygame.K_f:
                if flash_neon_prepare == 1:
                    neon_flash = 1 - neon_flash
                else:
                    neon_flash = 0
            elif event.key == pygame.K_1:
                major_key = 'C'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_2:
                major_key = 'Db'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_3:
                major_key = 'D'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_4:
                major_key = 'Eb'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_5:
                major_key = 'E'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_6:
                major_key = 'F'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_7:
                major_key = 'Gb'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_8:
                major_key = 'G'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_9:
                major_key = 'Ab'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_0:
                major_key = 'A'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_LEFT:
                major_key = 'Bb'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_RIGHT:
                major_key = 'B'
                major_key_print = font1.render(str('Majorkey: ' + major_key), True, major_key_text_color)
            elif event.key == pygame.K_e:
                for i in range(0, 88):
                    waterfalls[i].clear()
            elif event.key == pygame.K_q:
                if_exit = 1
                # 卸载所有模块
                pygame.quit()
                # 终止程序，确保退出程序
                sys.exit()
    pygame.display.flip()  # 更新屏幕内容
