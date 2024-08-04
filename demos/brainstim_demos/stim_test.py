import time
from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    paradigm_apply,
)
from paradigm_apply import Paradigm
from metabci.brainstim.framework import AssistBCI
import ctypes
import multiprocessing
from app_test import menu
import os
from app_test import stim_pos_setting, Experiment
from demos.brainstim_demos.device_worker import Device
from key_mouse_beta import CMD_Handler
import psutil

# if __name__ == "__main__":
#     dict = multiprocessing.Manager().dict()
#     dict['sys_key'] = None
#     dict['start_app'] = None
#     dict['start_par'] = None
#     dict['pause_app'] = False
#     dict['pra_exit_check'] = False
#     dict['save_par'] = False
#     dict['data_par'] = None
#     dict['app_exit_check'] = False
#     controller = CMD_Handler(dict)
#     controller.start()

def _get_status(subprocess=None):

    if subprocess._closed:
        return 'closed'
    if subprocess._popen is None:
        if not subprocess.is_alive():
            return 'initial'
    else:
        exitcode = subprocess._popen.poll()
        if exitcode is not None:
            exitcode = multiprocessing.process._exitcode_to_name.get(exitcode, exitcode)
            return 'stopped'
        else:
            if subprocess._parent_pid != os.getpid():
                return 'unknown'
            else:
                return 'started'


if __name__ == "__main__":

    #ctypes.windll.user32.SetProcessDPIAware()
    win_size = np.array([1920, 1081])


    dict = multiprocessing.Manager().dict()

    app_list = {'stim_pos_setting': stim_pos_setting, 'Experiment': Experiment}
    dict['app_list'] = [str(key) for key in app_list.keys()]
    dict['sys_key'] = None
    dict['start_app'] = None
    dict['start_par'] = None
    dict['pause_app'] = False
    dict['pra_exit_check'] = False
    dict['save_par'] = False
    dict['data_par'] = None
    dict['app_exit_check'] = False
    dict['framework_state'] = 'closed'
    dict['framework_type'] = 'application'
    dict['experiment_parameters'] = None

    app = {}
    for app_name in app_list.keys():
        app[app_name] = app_list[app_name](dict, win_size[0], win_size[1], 0, caption="PsychoPy", fullscreen=False)
    print("apps:", app)


    device = Device(dict)
    controller = CMD_Handler(dict)
    paradigm = Paradigm(dict, win_size=[win_size[0], win_size[1]], fps=60)
    Main = menu(dict, win_size[0], win_size[1], 0,
                  caption="PsychoPy",
                  fullscreen=False)

    controller.start()
    device.start()
    paradigm.start()
    Main.start()

    controller_pid = controller.pid
    device_pid = device.pid
    paradigm_pid = paradigm.pid
    Main_pid = Main.pid

    # speedup = psutil.Process(paradigm.pid)
    # speedup.nice(psutil.REALTIME_PRIORITY_CLASS)

    if dict["connect_device"] == None:
        dict['goto_menu'] = 'Desk/Device connection'
    while dict["device_state"] == 'not_connected':
        time.sleep(0.1)
    dict["reg_worker"] = 'ControlWorker'
    dict["start_worker"] = True

    time.sleep(10)
    dict['framework_state'] = 'hiding'
    # dict['start_par'] = '#menu'
    dict['goto_menu'] = 'Desk'



    while True:
        if dict['start_app'] != None:
            app_name = dict['start_app']
            print("start APP name:", app_name)
            app[app_name].start()
            dict["start_worker"] = True
            dict['app_exit_check'] = True
            dict['start_app'] = None
        # if dict['start_par'] != None:
        #     if dict['framework_state'] == 'started':
        #         dict['framework_state'] = 'hiding'


        if _get_status(paradigm) == 'stopped' or _get_status(paradigm) == 'closed':
            print("paradigm_finished")
            paradigm = Paradigm(dict=dict, win_size=[win_size[0], win_size[1]], fps=60)
            paradigm.start()

        # if not paradigm.is_alive():
        #     paradigm = Paradigm(dict=dict, win_size=[win_size[0], win_size[1]], fps=90)
        #     paradigm.start()

        if dict['app_exit_check']:   ## reload app
            if _get_status(app[app_name]) == 'stopped' or _get_status(app[app_name]) == 'closed':
                dict['app_exit_check'] = False
                app[app_name] = eval(app_name + '(dict, win_size[0], win_size[1], 0, caption="PsychoPy", fullscreen=False)')
                Main.restart_app()



