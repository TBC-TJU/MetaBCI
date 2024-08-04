# -*- coding: utf-8 -*-
import datetime
import gc
from collections import OrderedDict
from functools import partial
import numpy as np
from psychopy import core, visual, event, logging
from multiprocessing import Process, Lock ,Event, Queue, Manager
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import os
from .utils import _check_array_like, _clean_dict

class Experiment:
    """Paradigm start screen.

    author: Lichao Xu

    Created on: 2020-07-30

    update log:
        2022-08-10 by Wei Zhao

        2023-12-09 by Simiao Li <lsm_sim@tju.edu.cn> Add code annotation

    Parameters
    ----------
        win_size: tuple,shape(width,high)
            Size of the window in pixels [x, y].
        screen_id: int
            The id of screen. Specifies the physical screen on which the stimulus will appear;
            the default value is 0, and the value can be >0 if multiple screens are present.
        is_fullscr: bool
             Whether to create a window in 'full screen' mode.
        monitor: Monitor
             The monitor to be used during the experiment, if None the default monitor profile will be used.
        bg_color_warm: ndarray,shape(red,green,blue)
            The background color of the start screen, in [r, g, b] format, with values between -1.0 and 1.0.
        record_frames: bool
            Record time elapsed per frame, providing an accurate measurement of the frame interval
            to determine if the frame was discarded.
        disable_gc: bool
            Disable the garbage collector.
        process_priority: str
            Processing Priority.
        use_fbo: bool
            The FBO for a particular window can be switched for multi-window drawing,
            but is not needed in the general paradigm.

    Attributes
    ----------
        win_size: tuple,shape(width,high)
            Size of the window in pixels [x, y].
        screen_id: int
            The id of screen. Specifies the physical screen on which the stimulus will appear;
            the default value is 0, and the value can be >0 if multiple screens are present.
        is_fullscr: bool
             Whether to create a window in 'full screen' mode.
        monitor: Monitor
             The monitor to be used during the experiment, if None the default monitor profile will be used.
        bg_color_warm: ndarray,shape(red,green,blue)
            The background color of the start screen, in [r, g, b] format, with values between -1.0 and 1.0.
        record_frames: bool
            Record time elapsed per frame, providing an accurate measurement of the frame interval
            to determine if the frame was discarded.
        current_win: None
            If the display window does not exist, the window is created according to the initialization parameters.
        cache_stims: Dict
            Preserving the stimulus realization of the paradigm.
        paradigms: OrderedDict
            User-created paradigms that allow multiple paradigms to be created at the same time.
        current_paradigm: None
            The current opt-in paradigm.

    Tip
    ----
    .. code-block:: python
        :caption: An example of drawing the start screen

        from psychopy import monitors
        import numpy as np
        from brainstim.framework import Experiment
        mon = monitors.Monitor(
                name='primary_monitor',
                width=59.6, distance=60,
                verbose=False
            )
        mon.setSizePix([1920, 1080])     # Resolution of the monitor
        mon.save()
        bg_color_warm = np.array([0, 0, 0])
        win_size=np.array([1920, 1080])
        # press esc or q to exit the start selection screen
        ex = Experiment(
            monitor=mon,
            bg_color_warm=bg_color_warm, # background of paradigm selecting interface[-1~1,-1~1,-1~1]
            screen_id=0,
            win_size=win_size,           # Paradigm border size (expressed in pixels), default[1920,1080]
            is_fullscr=True,             # True full window, then win_size parameter defaults to the screen resolution
            record_frames=False,
            disable_gc=False,
            process_priority='normal',
            use_fbo=False)
        ex.register_paradigm(name, func, *args, **kwargs)

    See Also
    ----------
        _check_array_like(value, length=None)：
            Confirm the array dimension.
        _clean_dict(old_dict, includes=[])：
            Clear the dictionary.

    """

    def __init__(
        self,
        win_size=(800, 600),
        screen_id=0,
        is_fullscr=False,
        monitor=None,
        bg_color_warm=np.array([0, 0, 0]),
        record_frames=True,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    ):
        # global keys to exit experiment
        # only works in pyglet backend,the num lock key should be released first
        if not _check_array_like(win_size, 2):
            raise ValueError("win_size should be a 2 elements array-like object.")
        self.win_size = win_size

        if not isinstance(screen_id, int):
            raise ValueError("screen_id should be an int object")
        self.screen_id = screen_id

        if not _check_array_like(bg_color_warm, 3):
            raise ValueError("bg_color should be 3 elements array-like object.")
        self.bg_color_warm = bg_color_warm

        self.is_fullscr = is_fullscr
        self.monitor = monitor
        self.record_frames = record_frames

        # windows
        self.current_win = None

        # stimuli
        self.cache_stims = {}

        # paradigms
        self.paradigms = OrderedDict()
        self.current_paradigm = None

        #skip paradigm
        self.skip_par = False

        # high performance twicking
        visual.useFBO = use_fbo
        if process_priority == "normal":
            pass
        elif process_priority == "high":
            core.rush(True)
        elif process_priority == "realtime":
            # Only makes a diff compared to 'high' on Windows.
            core.rush(True, realtime=True)
        else:
            print(
                "Invalid process priority:",
                process_priority,
                "Process running at normal.",
            )
            process_priority = "normal"

        if disable_gc:
            gc.disable()

    def initEvent(self):
        """Init operations before run."""
        self.global_clock = core.Clock()
        logging.setDefaultClock(self.global_clock)
        logging.console.setLevel(logging.WARNING)
        self.log_file = logging.LogFile(
            "logLastRun.log", filemode="w", level=logging.DATA
        )

        logging.warning(
            "============start experiment at {}============".format(
                datetime.datetime.now()
            )
        )
        event.clearEvents()
        event.globalKeys.add(key="escape", func=self.closeEvent) #添加全局按键，可以随时退出程序

    def closeEvent(self):
        """Close operation after run."""
        logging.warning(
            "============end Experiemnt at {}============".format(
                datetime.datetime.now()
            )
        )
        # fixed sys.meta_path error
        _clean_dict(self.cache_stims)
        # restore gamma map
        core.quit()

    def register_paradigm(self, name, func, *args, **kwargs):
        """Create Paradigms, which allows multiple paradigms to be created at the same time.

        Parameters:
            name: str
                Paradigm name.
            func:
                Paradigm realization function.

        """
        # fixed supplied arguments
        self.paradigms[name] = partial(func, *args, **kwargs)

    def unregister_paradigm(self, name):
        """Clear the created paradigm with the name "name".

        Parameters:
            name:str
                Paradigm name.

        """
        # clean stims
        self.cache_stims[name] = None
        del self.cache_stims[name]

        # clean paradigms
        self.paradigms[name] = None
        del self.paradigms[name]

    def get_window(self, win_style=None):
        """If the display window does not exist, the window is created according to the initialization parameters.

        update log:
            2022-08-10 by Wei Zhao

        """
        if not self.current_win:
            self.current_win = visual.Window(
                # the only-one option in psychopy, pygame is deprecated and glfw has lots of bugs
                winType="pyglet",
                units="pix",  # default pixel unit in this framework
                allowGUI=False,
                win_style=win_style,
                # Here are timing related options
                waitBlanking=False,  # much faster
                useFBO=False,
                checkTiming=True,
                numSamples=2,
                # user specified options
                size=self.win_size,
                screen=self.screen_id,
                monitor=self.monitor,
                fullscr=self.is_fullscr,
                color=self.bg_color_warm,
                pos=[0, 0]
            )
            self.current_win.flip()
        return self.current_win

    def warmup(self, strict=True):
        """Set the window parameters further.

        """
        win = self.get_window()
        fps = win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1
        )  # keep int refresh rate
        if strict and fps is None:
            raise ValueError(
                "Can't get stable refresh rate. Close unnecessary programs or buy a better graphic cards."
            )

        self.fps = (
            int(np.rint(fps)) if fps is not None else int(1 / win.monitorFramePeriod)
        )
        self.frame_period = 1 / self.fps
        logging.warning(
            "Current screen refresh rate {}Hz and frame period {:.2f}ms".format(
                self.fps, self.frame_period * 1000
            )
        )
        win.recordFrameIntervals = self.record_frames
        win.refreshThreshold = 1 / self.fps + 0.002
        win.setMouseVisible(True)

    def update_startup(self):
        """Draw the start screen according to the custom paradigm and the stimulus implementation is saved in
        self.cache_stims.
2
        """
        win = self.get_window()
        #win.color = self.bg_color_warm
        back = visual.GratingStim
        stims = self.cache_stims.setdefault("startup", OrderedDict())

        # check cache stims
        if "expguide_textstim" not in stims:
            stims["expguide_textstim"] = visual.TextStim(
                win,
                text="Welcome to the BCI world!\nPress Enter to select one of the following paradigms\nPress q to "
                     "quit\n"
                "You can press esc to leave the program at any time!",
                units="height",
                pos=[0, 0.3],
                height=0.04,
                color="#ff944d",
                bold=False,
                alignText="center",
                anchorHoriz="center",
                wrapWidth=2,
            )

        # remove useless paradigms
        excludes = ["expguide_textstim"]
        stims = _clean_dict(stims, list(self.paradigms.keys()) + excludes)

        # update paradigm parameters
        names = list(self.paradigms.keys())
        if names and self.current_paradigm is None:
            self.current_paradigm = names[0]

        for i, name in enumerate(names):
            if name not in stims:
                stims[name] = visual.TextStim(
                    win,
                    text=name,
                    units="height",
                    pos=[0, -i * 0.03],
                    height=0.04,
                    color="#cccccc",
                    alignText="center",
                    anchorHoriz="center",
                    wrapWidth=1,
                )
            stims[name].setPos([0, -0.1 - i * 0.05])
            if name == self.current_paradigm:
                stims[name].setColor("#ff944d")
            else:
                stims[name].setColor("#cccccc")

        # draw all of them according to insert order
        for stim_name in stims:
            stims[stim_name].draw()

    def goto_par(self, par_name:str):
        self.current_paradigm = par_name
        self.skip_par = True

    def run(self):
        """Run the main loop."""
        self.initEvent()
        self.warmup()
        win = self.get_window()

        if self.record_frames:
            fps_textstim = visual.TextStim(
                win,
                text="",
                units="norm",
                pos=[-0.95, 0.95],
                height=0.03,
                color="#f2f2f2",
                alignText="left",
                anchorHoriz="left",
            )

        trialClock = core.Clock()
        t = lastFPSupdate = 0

        pindex = 0
        # capture runtime errors
        try:
            while True:
                t = trialClock.getTime()
                keys = event.getKeys(keyList=["q", "up", "down", "return"])

                # exit program
                if "q" in keys:
                    break

                # select paradigm
                names = list(self.paradigms.keys())
                if names:
                    if "up" in keys:
                        pindex -= 1
                        pindex = pindex % len(names)
                    elif "down" in keys:
                        pindex += 1
                        pindex = pindex % len(names)
                    self.current_paradigm = names[pindex]

                if "return" in keys or self.skip_par:
                    old_color = win.color
                    logging.warning("Start paradigm {}".format(self.current_paradigm))
                    self.paradigms[self.current_paradigm](win=win)
                    logging.warning("Finish paradigm {}".format(self.current_paradigm))
                    win.color = old_color

                if self.skip_par:
                    self.skip_par = False
                    break

                # main interface
                self.update_startup()

                if self.record_frames:
                    if t - lastFPSupdate > 1:
                        fps_textstim.text = "%i fps" % win.fps()
                        lastFPSupdate += 1
                    fps_textstim.draw()

                win.flip()

        except Exception as e:
            print("Error Info:", e)
            raise e
        finally:
            if self.record_frames:
                win.saveFrameIntervals("logLastFrameIntervals.log")
            win.close()
            print("end_____________")
            self.closeEvent()


class AssistBCI:

    def __init__(
        self,
        dict={},
        win_size=(800, 600),
        screen_id=0,
        is_fullscr=False,
        monitor=None,
        bg_color_warm=np.array([0, 0, 0]),
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    ):
        # global keys to exit experiment
        # only works in pyglet backend,the num lock key should be released first
        if not _check_array_like(win_size, 2):
            raise ValueError("win_size should be a 2 elements array-like object.")
        self.win_size = win_size

        if not isinstance(screen_id, int):
            raise ValueError("screen_id should be an int object")
        self.screen_id = screen_id

        if not _check_array_like(bg_color_warm, 3):
            raise ValueError("bg_color should be 3 elements array-like object.")
        self.bg_color_warm = bg_color_warm

        self.is_fullscr = is_fullscr
        self.monitor = monitor
        self.record_frames = record_frames

        # windows
        self.current_win = None

        # stimuli
        self.cache_stims = {}

        # paradigms
        self.paradigms = OrderedDict()
        self.current_paradigm = None

        self.lock = Lock()
        self._buffer = dict

        # high performance twicking
        visual.useFBO = use_fbo
        if process_priority == "normal":
            pass
        elif process_priority == "high":
            core.rush(True)
        elif process_priority == "realtime":
            # Only makes a diff compared to 'high' on Windows.
            core.rush(True, realtime=True)
        else:
            print(
                "Invalid process priority:",
                process_priority,
                "Process running at normal.",
            )
            process_priority = "normal"

        if disable_gc:
            gc.disable()

    def initEvent(self):
        """Init operations before run."""
        self.global_clock = core.Clock()
        logging.setDefaultClock(self.global_clock)
        logging.console.setLevel(logging.WARNING)
        self.log_file = logging.LogFile(
            "logLastRun.log", filemode="w", level=logging.DATA
        )

        logging.warning(
            "============start AssistBCI at {}============".format(
                datetime.datetime.now()
            )
        )

    def closeEvent(self):
        """Close operation after run."""
        logging.warning(
            "============end AssistBCI at {}============".format(
                datetime.datetime.now()
            )
        )
        # fixed sys.meta_path error
        _clean_dict(self.cache_stims)
        # restore gamma map
        core.quit()

    def register_paradigm(self, name, func, *args, **kwargs):
        """Create Paradigms, which allows multiple paradigms to be created at the same time.

        Parameters:
            name: str
                Paradigm name.
            func:
                Paradigm realization function.

        """
        # fixed supplied arguments
        self.paradigms[name] = partial(func, *args, **kwargs)

    def unregister_paradigm(self, name):
        """Clear the created paradigm with the name "name".

        Parameters:
            name:str
                Paradigm name.

        """
        # clean stims
        self.cache_stims[name] = None
        del self.cache_stims[name]

        # clean paradigms
        self.paradigms[name] = None
        del self.paradigms[name]

    def get_window(self, win_style=None, reload=False, win_size=None, screen_id=None, is_fullscr=None, bg_color_warm=None):
        """If the display window does not exist, the window is created according to the initialization parameters.

        update log:
            2022-08-10 by Wei Zhao

        """
        if reload:
            self.win_size = win_size
            self.screen_id = screen_id
            self.is_fullscr = is_fullscr
            self.bg_color_warm = bg_color_warm

        if not self.current_win or reload:
            print("inside")
            self.current_win = visual.Window(
                # the only-one option in psychopy, pygame is deprecated and glfw has lots of bugs
                winType="pyglet",
                units="pix",  # default pixel unit in this framework
                allowGUI=True,
                win_style=win_style,
                # Here are timing related options
                waitBlanking=False,  # much faster
                useFBO=False,
                checkTiming=True,
                numSamples=2,
                # user specified options
                size=self.win_size,
                screen=self.screen_id,
                monitor=self.monitor,
                fullscr=self.is_fullscr,
                color=self.bg_color_warm,
            )
            self.current_win.flip()
        return self.current_win

    def warmup(self, strict=True):
        """Set the window parameters further.

        """
        win = self.get_window(win_style='overlay')
        fps = win.getActualFrameRate(
            nIdentical=10, nMaxFrames=100, nWarmUpFrames=10, threshold=1
        )  # keep int refresh rate
        if strict and fps is None:
            raise ValueError(
                "Can't get stable refresh rate. Close unnecessary programs or buy a better graphic cards."
            )

        self.fps = (
            int(np.rint(fps)) if fps is not None else int(1 / win.monitorFramePeriod)
        )
        self.frame_period = 1 / self.fps
        logging.warning(
            "Current screen refresh rate {}Hz and frame period {:.2f}ms".format(
                self.fps, self.frame_period * 1000
            )
        )
        win.recordFrameIntervals = self.record_frames
        win.refreshThreshold = 1 / self.fps + 0.002
        win.setMouseVisible(True)


    def update_stim(self):
        win = self.get_window()
        # win.color = self.bg_color_warm
        back = visual.GratingStim
        stims = self.cache_stims.setdefault("startup", OrderedDict())
        # remove useless paradigms
        stims = _clean_dict(stims, list(self.paradigms.keys()))
        return win, back, stims

    def update_startup(self):
        """Draw the start screen according to the custom paradigm and the stimulus implementation is saved in
        self.cache_stims.

        """
        win, back, stims = self.update_stim()

        # update paradigm parameters
        names = [name for name in list(self.paradigms.keys()) if not name.startswith("#")]
        if names and self.current_paradigm is None:
            self.current_paradigm = names[0]

        for i, name in enumerate(names):
            if name not in stims and not name.startswith("#"):
                stims[name] = visual.TextStim(
                    win,
                    text=name,
                    units="height",
                    pos=[0, -i * 0.03],
                    height=0.04,
                    color="#cccccc",
                    alignText="center",
                    anchorHoriz="center",
                    wrapWidth=1,
                )
            stims[name].setPos([0, -0.1 - i * 0.05])
            if name == self.current_paradigm:
                stims[name].setColor("#ff944d")
            else:
                stims[name].setColor("#cccccc")

        # draw all of them according to insert order
        for stim_name in stims:
            stims[stim_name].draw()

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

    def list_files_in_stim_position(self, path: Optional[str] = ''):

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



    def run(self, save_path, file_paths):
        """Run the main loop."""
        if self.get("framework_state") == 'closed':
            self.send("framework_state", 'started')
            self.initEvent()
            self.warmup()
            # capture runtime errors
            trialClock = core.Clock()
            t = lastFPSupdate = 0
            win = self.get_window(win_style='overlay')
            if self.record_frames:
                fps_textstim = visual.TextStim(
                    win,
                    text="",
                    units="norm",
                    pos=[-0.95, 0.95],
                    height=0.03,
                    color="#f2f2f2",
                    alignText="left",
                    anchorHoriz="left",
                )

        try:

            self.send('quit_par', False)
            self.send("current_par", None)

            while True:
                t = trialClock.getTime()
                win = self.get_window(win_style='overlay')

                #更新范式需要退出
                changes = set(self.list_files_in_stim_position(save_path)).symmetric_difference(set(file_paths))
                if changes:
                    break

                framework_state = self.get("framework_state")

                if framework_state == 'showing':
                    pindex = 0
                    keys = event.getKeys(keyList=["q", "up", "down", "return"])
                    # exit program
                    if "q" in keys:
                        break

                    # select paradigm
                    names = [name for name in list(self.paradigms.keys()) if not name.startswith("#")]

                    if names:
                        if "up" in keys:
                            pindex -= 1
                            pindex = pindex % len(names)
                        elif "down" in keys:
                            pindex += 1
                            pindex = pindex % len(names)
                        self.current_paradigm = names[pindex]

                    if "return" in keys:
                        old_color = win.color
                        self.send("current_par", self.current_paradigm)
                        print("CMD_list:", self._buffer['CMD_list'][self._buffer["current_par"]])
                        logging.warning("Start paradigm {}".format(self.current_paradigm))
                        self.paradigms[self.current_paradigm](win=win)
                        logging.warning("Finish paradigm {}".format(self.current_paradigm))
                        self.send("current_par", None)
                        win.color = old_color

                    elif self.get("start_par") != None:
                        paradigm = self.get("start_par")
                        self.send("start_par", None)
                        old_color = win.color
                        self.send("current_par", paradigm)
                        print("CMD_list:", self._buffer['CMD_list'][self._buffer["current_par"]])
                        logging.warning("Start paradigm {}".format(self.current_paradigm))
                        self.paradigms[paradigm](win=win)
                        logging.warning("Finish paradigm {}".format(self.current_paradigm))
                        self.send("current_par", None)
                        win.color = old_color

                    # main interface
                    self.update_startup()

                elif framework_state == 'hiding':
                    if self.get("start_par") != None:
                        paradigm = self.get("start_par")
                        self.send("start_par", None)
                        old_color = win.color
                        self.send("current_par", paradigm)
                        #print("CMD_list:", self._buffer['CMD_list'][self._buffer["current_par"]])
                        logging.warning("Start paradigm {}".format(self.current_paradigm))
                        self.paradigms[paradigm](win=win)
                        logging.warning("Finish paradigm {}".format(self.current_paradigm))
                        self.send("current_par", None)
                        win.color = old_color

                elif framework_state == 'closed':
                    break

                elif framework_state == 'started': #待机模式
                    pass

                if self.record_frames:
                    if t - lastFPSupdate > 1:
                        fps_textstim.text = "%i fps" % win.fps()
                        lastFPSupdate += 1
                    fps_textstim.draw()


                win.flip()

        except Exception as e:
            print("Error Info:", e)
            framework_state = 'closed'
            self.send('framework_state', framework_state)
            raise e

        finally:
            if framework_state == 'closed':
                print("closed")
                if self.record_frames:
                    win.saveFrameIntervals("logLastFrameIntervals.log")
                win.close()
                self.closeEvent()

