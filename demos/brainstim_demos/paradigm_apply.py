import os.path
from abc import abstractmethod
from typing import Optional, Any
from typing import Union, Optional, Dict, List, Tuple
from multiprocessing import Process, Lock ,Event, Queue, Manager
import threading
import numpy as np
from pathlib import Path
from metabci.brainflow.amplifiers import NeuroScan, BlueBCI, Curry8, Neuracle
from metabci.brainflow.amplifiers import Marker
from functools import partial
from workers import BasicWorker, ControlWorker
import time
import ctypes
from psychopy import monitors
from metabci.brainstim.framework import AssistBCI, Experiment
from metabci.brainstim.paradigm import (
    SSVEP,
    paradigm_apply,
    paradigm,
)


class Paradigm(Process):

    def __init__(self, dict, timeout: float = 5, win_size: list=[1920,1080], fps: int=90, path='', experiment=False):
        Process.__init__(self)

        self.timeout = timeout
        self.lock = Lock()
        self._buffer = dict
        self.fps = fps
        self.win_size = win_size
        self.win_size_par = win_size
        self.path = path
        self.file_paths = []
        self.paradigm_nameList = []
        self.key_mouse_mapping = {}
        self.send('paradigm_list', [])
        self.send('CMD_list', {})
        # self.file_paths = self.list_files_in_stim_position(self.path)

    def save_hyper(self, *args, **kwargs):
        return args, kwargs

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
            self._buffer[name+'_arg'] = arg
            self._buffer[name + '_kwargs'] = kwargs
        finally:
            # 无论如何都要释放锁
            self.lock.release()

    def get(self, name):
        return self._buffer[name]

    def get_hyper(self, name):
        return self._buffer[name+'_arg'], self._buffer[name + '_kwargs']


    def start_framework(self, framework='application'):
        ctypes.windll.user32.SetProcessDPIAware()
        print("start_framework")
        #self.send('paradigm_list', [])
        self.mon = self.monitor_setup(self.win_size)
        bg_color_warm = np.array([0.3, 0.3, 0.3])
        if framework != 'experiment':
            self.framework = AssistBCI(
                dict=self._buffer,
                monitor=self.mon,
                bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
                screen_id=0,
                win_size=self.win_size,  # 范式边框大小(像素表示)，默认[1920,1080]
                is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
                record_frames=True,
                disable_gc=False,
                process_priority="normal",
                use_fbo=False,
            )
            self.win = self.framework.get_window(win_style='overlay')
        elif framework == 'experiment':
            self.framework = Experiment(
                monitor=self.mon,
                bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
                screen_id=0,
                win_size=self.win_size_par,  # 范式边框大小(像素表示)，默认[1920,1080]
                is_fullscr=False,  # True全窗口,此时win_size参数默认屏幕分辨率
                record_frames=False,
                disable_gc=False,
                process_priority="normal",
                use_fbo=False,
            )
            self.win = self.framework.get_window()

    def list_files_in_stim_position(self, path: Optional[str] = ''):
        #Eg. path = 'C:\\Users\\abc\\AssistBCI'
        if path == '':
            home_dir = Path.home()  # 获取用户主目录
            self.stim_position_path = home_dir / "AssistBCI" / "stim_position"  # 构建目标路径

            if not os.path.exists(self.stim_position_path):
                os.makedirs(self.stim_position_path)
        else:
            self.stim_position_path = path

        # 列出所有文件
        file_paths = [file for file in self.stim_position_path.glob('*') if file.is_file()]

        return file_paths


    def read_file(self, filename):
        try:
            # 读取txt文件
            with open(filename, 'r') as f:
                lines = f.readlines()
                # 解析文件内容
                stim_info = {}
                for line in lines:
                    elements = line.split('\n')[0]
                    elements = elements.split('=')
                    try:
                        stim_info[elements[0]] = eval('='.join(elements[1:]))
                    except:
                        stim_info[elements[0]] = '='.join(elements[1:])
                return stim_info

        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return []

    def monitor_setup(self, win_size=[1920,1080], width=59.6, distance=60):
        mon = monitors.Monitor(
            name="primary_monitor",
            width=width,
            distance=distance,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
            verbose=False,
        )
        mon.setSizePix(win_size)  # 显示器的分辨率
        mon.save()
        return mon

    def _validity_test(self, stim_info):
        ssvep_keys1 = ['paradigm', 'name', 'stim_names', 'stim_pos', 'n_elements',
                       'stim_length', 'stim_width','stim_time', 'freqs','phases', 'key_mouse_mapping']
        ssvep_keys2 = ['paradigm', 'name', 'stim_names', 'rows', 'columns', 'n_elements',
                       'stim_length', 'stim_width','stim_time', 'freqs','phases', 'key_mouse_mapping']

        try:
            if 'paradigm' in stim_info.keys():
                paradigm = stim_info['paradigm']
                if paradigm == 'ssvep':
                    for req_key1 in ssvep_keys1:
                        if req_key1 not in stim_info.keys():
                            print(req_key1, "not find in file")
                            for req_key2 in ssvep_keys2:
                                if req_key2 not in stim_info.keys():
                                    print(req_key2, "not find in file")
                                    return False
            else:
                print("can not find Key: paradigm")
                return False
        except:
            return False


        # 检查值是否符合特定格式....

        # 如果所有检查通过，返回True表示格式正确
        return True

    def ssvep(self, stim_info: dict, framework_type='application'):
        """
        info中需要包含：
        paradigm, name, stim_names, stim_pos(or rows and columns),
        n_elements，stim_length, stim_width

        experiment:
        paradigm, rows and columns, n_elements，stim_length, stim_width, rest_time
        display_time, index_time, response_time, port_addr, nrep, online, device_type("Light_trigger"...)
            """
        stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
        stim_opacities = 1  # 刺激对比度
        bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色

        #影响性能，不开放自定义
        stim_time = 6.0
        freqs = np.array(stim_info['freqs'])
        phases = np.array(stim_info['phases'])

        if framework_type == 'application':
            rest_time = 0
        elif framework_type == 'experiment':
            lsl_source_id = "meta_online_worker"

        basic_ssvep = SSVEP(win=self.win)
        print('freqs:', freqs)

        if 'stim_pos' in stim_info.keys():
            basic_ssvep.config_pos(
                n_elements=stim_info['n_elements'],
                stim_length=stim_info['stim_length'],
                stim_width=stim_info['stim_width'],
                stim_pos=np.array(stim_info['stim_pos']),
            )
        else:
            basic_ssvep.config_pos(
                n_elements=stim_info['n_elements'],
                stim_length=stim_info['stim_length'],
                stim_width=stim_info['stim_width'],
                rows=stim_info['rows'],
                columns=stim_info['columns'],
            )

        if 'stim_names' in stim_info.keys():
            basic_ssvep.config_text(
                symbols=stim_info['stim_names'],
                tex_color=tex_color)
        else:
            basic_ssvep.config_text(tex_color=tex_color)

        basic_ssvep.config_color(
            refresh_rate=self.fps,
            stim_time=stim_time,
            stimtype="sinusoid",
            stim_color=stim_color,
            stim_opacities=stim_opacities,
            freqs=freqs,
            phases=phases,
        )
        basic_ssvep.config_index()

        basic_ssvep.config_response()

        if framework_type == 'application':
            self.framework.register_paradigm(
                stim_info['name'],
                paradigm_apply,
                VSObject=basic_ssvep,
                dict=self._buffer,
                bg_color=bg_color,
                rest_time=rest_time,
                pdim="ssvep",
            )
        elif framework_type == 'experiment':
            self.framework. register_paradigm(
                "Training SSVEP",
                paradigm,
                VSObject=basic_ssvep,
                bg_color=bg_color,
                display_time=stim_info['display_time'],
                index_time=stim_info['index_time'],
                rest_time=stim_info['rest_time'],
                response_time=stim_info['response_time'],
                port_addr=stim_info['port_addr'],
                nrep=stim_info['nrep'],
                pdim="ssvep",
                lsl_source_id=lsl_source_id,
                online=stim_info['online'],
                device_type=stim_info['device_type'],
                w=self.win_size[0],
                h=self.win_size[1],
            )



    def run(self):
        framework_type = self.get('framework_type')
        print("type:", framework_type)
        first_time = True

        if framework_type == 'application':
            self.start_framework(framework=framework_type)
            while self.get("framework_state") != 'closed' or first_time:
                first_time = False
                changes = set(self.list_files_in_stim_position(self.path)).symmetric_difference(set(self.file_paths))
                for path in changes:
                    if path in self.file_paths:
                        print("path:", path)
                        name = str(os.path.splitext(os.path.split(path)[1])[0])
                        print("name:", name)
                        self.framework.unregister_paradigm(name)
                        self.file_paths.remove(path)
                        self.paradigm_nameList.remove(name)
                        del self.key_mouse_mapping[name]
                    else:
                        stim_info = self.read_file(path)
                        if not self._validity_test(stim_info):
                            print("invalidate file:", str(path))
                            continue
                        if stim_info['paradigm'] == 'ssvep':
                            self.ssvep(stim_info, framework_type='application')
                            self.paradigm_nameList.append(stim_info['name'])
                            self.key_mouse_mapping[stim_info['name']] = stim_info['key_mouse_mapping']
                        self.file_paths.append(path)
                self.send('paradigm_list', self.paradigm_nameList)
                self.send('CMD_list', self.key_mouse_mapping)
                print("---------------------start framework----------------------")
                self.framework.run(save_path=self.path, file_paths=self.file_paths)

        elif framework_type == 'experiment':
            while self.get('experiment_parameters') == None:
                time.sleep(0.1)
            stim_info = self.get('experiment_parameters')
            self.send('experiment_parameters', None)

            print("stim_info['paradigm']=", stim_info['paradigm'])

            if stim_info['device_type'] == "Light_trigger":
                self.win_size_par = [self.win_size[0], int(self.win_size[1]*(7/8))]
            self.start_framework(framework=framework_type)

            self.send('framework_type', 'application')
            if stim_info['paradigm'] == 'SSVEP':
                self.ssvep(stim_info, framework_type='experiment')
                self.framework.goto_par("Training SSVEP")
                print("set goto_par OK")
                self.send('paradigm_list', [])
                self.send('CMD_list', {})
                self.framework.run()
                print("here1")


