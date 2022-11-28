# -*- coding: utf-8 -*-

# load in basic modules
import os
import string
import numpy as np
from math import pi
from psychopy import data, visual, event
from pylsl.pylsl import StreamInlet, resolve_byprop
from .utils import NeuroScanPort, _check_array_like
import threading
from copy import copy

# prefunctions


def sinusoidal_sample(freqs, phases, srate, frames, stim_color):
    """Sinusoidal approximate sampling method.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
        freqs: list of float,
            Frequencies of each stimulus.
        phases: list of float,
            Phases of each stimulus.
        srate: int or float,
            Refresh rate of screen.
        frames: int,
            Flashing frames.
        stim_color: list,
            Color of stimu.
    Returns:
    ----------
        color: ndarray,
            (n_frames, n_elements, 3)
    """

    time = np.linspace(0, (frames - 1) / srate, frames)
    color = np.zeros((frames, len(freqs), 3))
    for ne, (freq, phase) in enumerate(zip(freqs, phases)):
        sinw = np.sin(2 * pi * freq * time + pi * phase) + 1
        color[:, ne, :] = np.vstack(
            (sinw * stim_color[0], sinw * stim_color[1], sinw * stim_color[2])
        ).T
        if stim_color == [-1, -1, -1]:
            pass
        else:
            if stim_color[0] == -1:
                color[:, ne, 0] = -1
            if stim_color[1] == -1:
                color[:, ne, 1] = -1
            if stim_color[2] == -1:
                color[:, ne, 2] = -1

    return color


# create interface for VEP-BCI-Speller


class KeyboardInterface(object):
    """Create stimulus interface.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        self.win = win
        win.colorSpace = colorSpace
        win.allowGUI = allowGUI
        win_size = win.size
        self.win_size = np.array(win_size)  # e.g. [1920,1080]

    def config_pos(
        self,
        n_elements=40,
        rows=5,
        columns=8,
        stim_pos=None,
        stim_length=150,
        stim_width=150,
    ):
        """Config positions of stimuli.
        -update log:
            2022-06-26 by Jianhang Wu
        Parameters
        ----------
            n_elements: int,
                Number of stimuli.
            rows: int, optional,
                Rows of keyboard.
            columns: int, optional,
                Columns of keyboard.
            stim_pos: ndarray, optional,
                Extra position matrix.
            stim_length: int,
                Length of stimulus.
            stim_width: int,
                Width of stimulus.
        Raises
        ----------
            Exception: Inconsistent numbers of stimuli and positions.
        """

        self.stim_length = stim_length
        self.stim_width = stim_width
        self.n_elements = n_elements
        # highly customizable position matrix
        if (stim_pos is not None) and (self.n_elements == stim_pos.shape[0]):
            # note that the origin point of the coordinate axis should be the center of your screen
            # (so the upper left corner is in Quadrant 2nd), and the larger the coordinate value,
            # the farther the actual position is from the center
            self.stim_pos = stim_pos
        # conventional design method
        elif (stim_pos is None) and (rows * columns >= self.n_elements):
            # according to the given rows of columns, coordinates will be automatically converted
            stim_pos = np.zeros((self.n_elements, 2))
            # divide the whole screen into rows*columns' blocks, and pick the center of each block
            first_pos = (
                np.array([self.win_size[0] / columns, self.win_size[1] / rows]) / 2
            )
            if (first_pos[0] < stim_length / 2) or (first_pos[1] < stim_width / 2):
                raise Exception("Too much blocks or too big the stimulus region!")
            for i in range(columns):
                for j in range(rows):
                    stim_pos[i * rows + j] = first_pos + [i, j] * first_pos * 2
            # note that those coordinates are still not the real ones that need to be set on the screen
            stim_pos -= self.win_size / 2  # from Quadrant 1st to 3rd
            stim_pos[:, 1] *= -1  # invert the y-axis
            self.stim_pos = stim_pos
        else:
            raise Exception("Incorrect number of stimulus!")

        # check size of stimuli
        stim_sizes = np.zeros((self.n_elements, 2))
        stim_sizes[:] = np.array([stim_length, stim_width])
        self.stim_sizes = stim_sizes
        self.stim_width = stim_width

    def config_text(self, symbols=None, symbol_height=0, tex_color=[1, 1, 1]):
        """Config text stimuli.
        -update log:
            2022-06-26 by Jianhang Wu
        Parameters
        ----------
            symbols: list of str,
                Target characters.
            symbol_height: int,
                Height of target symbol.
            tex_color: list,
                Color of target symbol.
        Raises:
            Exception: Insufficient characters.
        """

        # check number of symbols
        if (symbols is not None) and (len(symbols) >= self.n_elements):
            self.symbols = symbols
        elif self.n_elements <= 40:
            self.symbols = "".join([string.ascii_uppercase, "1234567890+-*/"])
        else:
            raise Exception("Please input correct symbol list!")

        # add text targets onto interface
        if symbol_height == 0:
            symbol_height = self.stim_width / 2
        self.text_stimuli = []
        for symbol, pos in zip(self.symbols, self.stim_pos):
            self.text_stimuli.append(
                visual.TextStim(
                    win=self.win,
                    text=symbol,
                    font="Times New Roman",
                    pos=pos,
                    color=tex_color,
                    units="pix",
                    height=symbol_height,
                    bold=True,
                    name=symbol,
                )
            )

    def config_response(
        self,
        symbol_text="Speller:  ",
        symbol_height=0,
        symbol_color=(1, 1, 1),
        bg_color=[-1, -1, -1],
    ):
        """Config response stimuli.
        -update log:
            2022-08-10 by Wei Zhao
        Parameters
        ----------
            symbol_text: list of str,
                Online response string.
            symbol_height: int,
                Height of response symbol.
            symbol_color: list,
                Color of response symbol.
            bg_color: list,
                Color of background symbol.
        Raises:
            Exception: Insufficient characters.
        """

        brige_length = self.win_size[0] / 2 + self.stim_pos[0][0] - self.stim_length / 2
        brige_width = self.win_size[1] / 2 - self.stim_pos[0][1] - self.stim_width / 2

        self.rect_response = visual.Rect(
            win=self.win,
            units="pix",
            width=self.win_size[0] - brige_length,
            height=brige_width * 3 / 4,
            pos=(0, self.win_size[1] / 2 - brige_width / 2),
            fillColor=bg_color,
            lineColor=[1, 1, 1],
        )

        self.res_text_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_pos = (
            -self.win_size[0] / 2 + brige_length * 3 / 2,
            self.win_size[1] / 2 - brige_width / 2,
        )
        self.reset_res_text = "Speller:  "
        if symbol_height == 0:
            self.symbol_height = brige_width / 2
        self.symbol_text = symbol_text
        self.text_response = visual.TextStim(
            win=self.win,
            text=symbol_text,
            font="Times New Roman",
            pos=self.res_text_pos,
            color=symbol_color,
            units="pix",
            height=self.symbol_height,
            bold=True,
        )


# config visual stimuli


class VisualStim(KeyboardInterface):
    """Create various visual stimuli.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)
        self._exit = threading.Event()

    def config_index(self, index_height=0):
        """Config index stimuli: downward triangle (Unicode: \u2BC6)
        Parameters
        ----------
            index_height: int, optional,
                Defaults to 75 pixels.
        """

        # add index onto interface, with positions to be confirmed.
        if index_height == 0:
            index_height = copy(self.stim_width / 3 * 2)
        self.index_stimuli = visual.TextStim(
            win=self.win,
            text="\u2BC6",
            font="Arial",
            color=[1.0, 1.0, 0.0],
            colorSpace="rgb",
            units="pix",
            height=index_height,
            bold=True,
            autoLog=False,
        )


# standard SSVEP paradigm


class SSVEP(VisualStim):
    """Create SSVEP stimuli.
    -author: Qiaoyi Wu
    -Created on: 2022-06-20
    -update log:
        2022-06-26 by Jianhang Wu
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        """Item class from VisualStim.

        Args:

        """
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(
        self,
        refresh_rate,
        stim_time,
        stim_color,
        stimtype="sinusoid",
        stim_opacities=1,
        **kwargs
    ):
        """Config color of stimuli.
        Parameters
        ----------
            refresh_rate: int or float,
                Refresh rate of screen.
            stim_time: float,
                Time of each stimulus.
            stim_frames: int,
                Flash frames of one trial.
            stim_colors: ndarray,
                (n_frames, n_elements, 3).
            stim_opacities: int or float,
                Opacities of each stimulus.
            freqs: list of float,
                Frequencies of each stimulus.
            phases: list of float,
                Phases of each stimulus.
        Raises:
            Exception: Inconsistent frames and color matrices.
        """

        # initialize extra inputs
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)

        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = np.ones((self.n_elements,))  # contrast

        # check extra inputs
        if "stim_oris" in kwargs.keys():
            self.stim_oris = kwargs["stim_oris"]
        if "stim_sfs" in kwargs.keys():
            self.stim_sfs = kwargs["stim_sfs"]
        if "stim_contrs" in kwargs.keys():
            self.stim_contrs = kwargs["stim_contrs"]
        if "freqs" in kwargs.keys():
            self.freqs = kwargs["freqs"]
        if "phases" in kwargs.keys():
            self.phases = kwargs["phases"]

        # check consistency
        if stimtype == "sinusoid":
            self.stim_colors = sinusoidal_sample(
                freqs=self.freqs,
                phases=self.phases,
                srate=self.refresh_rate,
                frames=self.stim_frames,
                stim_color=stim_color,
            )
            if self.stim_colors[0].shape[0] != self.n_elements:
                raise Exception("Please input correct num of stims!")

        incorrect_frame = self.stim_colors.shape[0] != self.stim_frames
        incorrect_number = self.stim_colors.shape[1] != self.n_elements
        if incorrect_frame or incorrect_number:
            raise Exception("Incorrect color matrix or flash frames!")

        # add flashing targets onto interface
        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,
                    units="pix",
                    nElements=self.n_elements,
                    sizes=self.stim_sizes,
                    xys=self.stim_pos,
                    colors=self.stim_colors[sf, ...],
                    opacities=self.stim_opacities,
                    oris=self.stim_oris,
                    sfs=self.stim_sfs,
                    contrs=self.stim_contrs,
                    elementTex=np.ones((64, 64)),
                    elementMask=None,
                    texRes=48,
                )
            )


# standard P300 paradigm


class P300(VisualStim):
    """Create P300 stimuli.
    -author: Shengfu Wen
    -Created on: 2022-07-04
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_color(self, refresh_rate=0, stim_duration=0.5):
        """Config color of stimuli.
        Parameters
        ----------
            refresh_rate: int or float,
                Refresh rate of screen.
            symbol_height: float,
                Height of each stimulus.
            stim_duration: float,
                The duration of one trial.
        """
        self.stim_duration = stim_duration
        self.refresh_rate = refresh_rate
        if refresh_rate == 0:
            self.refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        # highlight one row/ column onto interface
        row_pos = np.unique(self.stim_pos[:, 0])
        col_pos = np.unique(self.stim_pos[:, 1])
        [row_num, col_num] = [len(col_pos), len(row_pos)]
        # complete single trial
        self.stim_frames = int((row_num + col_num) * stim_duration * refresh_rate)

        row_order_index = list(range(0, row_num))
        np.random.shuffle(row_order_index)
        col_order_index = list(range(0, col_num))
        np.random.shuffle(col_order_index)
        l_row_order_index = [
            x + self.n_elements + 1 for x in row_order_index
        ]  # reset event label
        l_col_order_index = [x + self.n_elements + row_num + 1 for x in col_order_index]

        self.order_index = np.array(
            l_row_order_index + l_col_order_index
        )  # event label

        # Determine row and column char status
        stim_colors_row = np.zeros(
            [(row_num * col_num), int(row_num * refresh_rate * stim_duration), 3]
        )
        stim_colors_col = np.zeros(
            [(row_num * col_num), int(col_num * refresh_rate * stim_duration), 3]
        )  #

        tmp = 0
        for col_i in col_order_index:
            stim_colors_col[
                (col_i * row_num) : ((col_i + 1) * row_num),
                int(tmp * refresh_rate * stim_duration) : int(
                    (tmp + 1) * refresh_rate * stim_duration
                ),
            ] = [-1, -1, -1]
            tmp += 1

        tmp = 0
        for row_i in row_order_index:
            for col_i in range(col_num):
                stim_colors_row[
                    (row_i + row_num * col_i),
                    int(tmp * refresh_rate * stim_duration) : int(
                        (tmp + 1) * refresh_rate * stim_duration
                    ),
                ] = [-1, -1, -1]
            tmp += 1

        stim_colors = np.concatenate((stim_colors_row, stim_colors_col), axis=1)
        self.stim_colors = np.transpose(stim_colors, [1, 0, 2])

        # add flashing targets onto interface
        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,
                    units="pix",
                    nElements=self.n_elements,
                    sizes=self.stim_sizes,
                    xys=self.stim_pos,
                    colors=self.stim_colors[sf, ...],
                    elementTex=np.ones((64, 64)),
                    elementMask=None,
                    texRes=48,
                )
            )


# standard MI paradigm


class MI(VisualStim):
    """Create MI stimuli.
    -author: Wei Zhao
    -Created on: 2022-06-30
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    win:
        The window object.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    """

    def __init__(self, win, colorSpace="rgb", allowGUI=True):
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

        self.tex_left = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures"+os.sep+"left_hand.png",
        )
        self.tex_right = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures"+os.sep+"right_hand.png",
        )

    def config_color(
        self,
        refresh_rate=60,
        text_pos=(0.0, 0.0),
        left_pos=[[-480, 0.0]],
        right_pos=[[480, 0.0]],
        tex_color=(1, -1, -1),
        normal_color=[[-0.8, -0.8, 0.8]],
        image_color=[[1, 1, 1]],
        symbol_height=100,
        n_Elements=1,
        stim_length=288,
        stim_width=162,
    ):
        """Config color of stimuli.
        Parameters
        ----------
            refresh_rate: int or float,
                Refresh rate of screen.
            text_pos: ndarray,
                Position of text.
            left_pos: list,
                Position of left hand.
            right_pos: list,
                Position of right hand.
            text_color: list,
                Color of text.
            normal_color: list,
                Color of default stimulus
            image_color: list,
                Color of image or indicate stimulus.
            symbol_height: list,
                Height of text.
            n_Elements: list,
                Num of stimulus.
            stim_length: list,
                Length of stimulus.
            stim_width: list,
                Width of stimulus.
        """

        self.n_Elements = n_Elements
        self.stim_length = stim_length
        self.stim_width = stim_width
        self.left_pos = left_pos
        self.right_pos = right_pos
        self.refresh_rate = refresh_rate
        if refresh_rate == 0:
            refresh_rate = np.floor(
                self.win.getActualFrameRate(nIdentical=20, nWarmUpFrames=20)
            )

        if symbol_height == 0:
            symbol_height = int(self.win_size[1] / 6)
        self.text_stimulus = visual.TextStim(
            self.win,
            text="start",
            font="Times New Roman",
            pos=text_pos,
            color=tex_color,
            units="pix",
            height=symbol_height,
            bold=True,
        )

        self.image_left_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_left,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(left_pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )
        self.image_right_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_right,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(right_pos),
            oris=[0],
            colors=np.array(image_color),
            opacities=[1],
            contrs=[-1],
        )

        self.normal_left_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_left,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(left_pos),
            oris=[0],
            colors=np.array(normal_color),
            opacities=[1],
            contrs=[-1],
        )
        self.normal_right_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_right,
            elementMask=None,
            texRes=2,
            nElements=n_Elements,
            sizes=[[stim_length, stim_width]],
            xys=np.array(right_pos),
            oris=[0],
            colors=np.array(normal_color),
            opacities=[1],
            contrs=[-1],
        )

    def config_response(self, response_color=[[-0.5, 0.9, 0.5]]):
        self.response_left_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_left,
            elementMask=None,
            texRes=2,
            nElements=self.n_Elements,
            sizes=[[self.stim_length, self.stim_width]],
            xys=np.array(self.left_pos),
            oris=[0],
            colors=np.array(response_color),
            opacities=[1],
            contrs=[-1],
        )
        self.response_right_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.tex_right,
            elementMask=None,
            texRes=2,
            nElements=self.n_Elements,
            sizes=[[self.stim_length, self.stim_width]],
            xys=np.array(self.right_pos),
            oris=[0],
            colors=np.array(response_color),
            opacities=[1],
            contrs=[-1],
        )


# continuous experimen


class GetPlabel_MyTherad:
    """Start a thread that receives online results
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
    Parameters
    ----------
    inlet:
        Stream data online.
    """

    def __init__(self, inlet):
        self.inlet = inlet
        self._exit = threading.Event()

    def feedbackThread(self):
        """Start the thread."""
        self._t_loop = threading.Thread(
            target=self._inner_loop, name="get_predict_id_loop"
        )
        self._t_loop.start()

    def _inner_loop(self):
        """The inner loop in the thread."""
        self._exit.clear()
        global online_text_pos, online_symbol_text
        online_text_pos = copy(self.res_text_pos)
        online_symbol_text = copy(self.symbol_text)
        while not self._exit.is_set():
            try:
                samples, _ = self.inlet.pull_sample()
                if samples:
                    # online predict id
                    predict_id = int(samples[0]) - 1
                    online_text_pos = (
                        online_text_pos[0] + self.symbol_height / 3,
                        online_text_pos[1],
                    )
                    online_symbol_text = online_symbol_text + self.symbols[predict_id]
            except Exception:
                pass

    def stop_feedbackThread(self):
        """Stop the thread."""
        self._exit.set()
        self._t_loop.join()


# basic experiment control


def paradigm(
    VSObject,
    win,
    bg_color,
    display_time=1.0,
    index_time=1.0,
    rest_time=0.5,
    response_time=2,
    image_time=2,
    port_addr=9045,
    nrep=1,
    pdim="ssvep",
    lsl_source_id=None,
    online=None,
):
    """Passing outsied parameters to inner attributes.
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
        2022-08-03 by Shengfu Wen
    Parameters
    ----------
        bg_color: ndarray,
            Background color.
        display_time: float,
            Keyboard display time before 1st index.
        index_time: float,
            Indicator display time.
        rest_time: float, optional,
            Rest-state time.
        respond_time: float, optional,
            Feedback time during online experiment.
        image_time: float, optional,
            Image time.
        port_addr:
             Computer port.
        nrep: int,
            Num of blocks.
        mi_flag: bool,
            Flag of MI paradigm.
        lsl_source_id: str,
            Source id.
        online: bool,
            Flag of online experiment.
    """

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
    port_frame = int(0.05 * fps)

    inlet = False
    if online:
        if pdim == "ssvep" or pdim == "p300" or pdim == "con-ssvep":
            VSObject.text_response.text = copy(VSObject.reset_res_text)
            VSObject.text_response.pos = copy(VSObject.reset_res_pos)
            VSObject.res_text_pos = copy(VSObject.reset_res_pos)
            VSObject.symbol_text = copy(VSObject.reset_res_text)
            res_text_pos = VSObject.reset_res_pos
        if lsl_source_id:
            inlet = True
            streams = resolve_byprop(
                "source_id", lsl_source_id, timeout=5
            )  # Resolve all streams by source_id
            if not streams:
                return
            inlet = StreamInlet(streams[0])  # receive stream data

    if pdim == "ssvep":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "p300":
        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if iframe == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, 1)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if iframe == port_frame and port:
                    port.setData(0)
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if (sf % (VSObject.stim_duration * fps)) == 0 and port:
                    VSObject.win.callOnFlip(
                        port.setData,
                        VSObject.order_index[int(sf / (VSObject.stim_duration * fps))],
                    )
                if (sf % (VSObject.stim_duration * fps)) == port_frame and port:
                    port.setData(0)

                VSObject.flash_stimuli[sf].draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()
                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id
                VSObject.symbol_text = (
                    VSObject.symbol_text + VSObject.symbols[predict_id]
                )
                res_text_pos = (
                    res_text_pos[0] + VSObject.symbol_height / 3,
                    res_text_pos[1],
                )
                iframe = 0
                while iframe < int(fps * response_time):
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = VSObject.symbol_text
                    VSObject.text_response.pos = res_text_pos
                    VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "mi":
        # config experiment settings
        conditions = [
            {"id": 0, "name": "left_hand"},
            {"id": 1, "name": "right_hand"},
            {"id": 2, "name": "both_hands"},
        ]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            VSObject.normal_left_stimuli.draw()
            VSObject.normal_right_stimuli.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                break

            # initialise index position
            id = int(trial["id"])
            if id == 0:
                image_stimuli = [VSObject.image_left_stimuli]
                normal_stimuli = [VSObject.normal_right_stimuli]
            elif id == 1:
                image_stimuli = [VSObject.image_right_stimuli]
                normal_stimuli = [VSObject.normal_left_stimuli]
            else:
                image_stimuli = [
                    VSObject.image_left_stimuli,
                    VSObject.image_right_stimuli,
                ]
                normal_stimuli = []

            # phase I: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    VSObject.normal_left_stimuli.draw()
                    VSObject.normal_right_stimuli.draw()
                    iframe += 1
                    win.flip()

            # phase II: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                for _image_stimuli in image_stimuli:
                    _image_stimuli.draw()
                if normal_stimuli:
                    for _normal_stimuli in normal_stimuli:
                        _normal_stimuli.draw()
                iframe += 1
                win.flip()

            # phase III: target stimulating
            iframe = 0
            while iframe < int(fps * image_time):
                if iframe == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, 1)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if iframe == port_frame and port:
                    port.setData(0)
                VSObject.text_stimulus.draw()
                for _image_stimuli in image_stimuli:
                    _image_stimuli.draw()
                if normal_stimuli:
                    for _normal_stimuli in normal_stimuli:
                        _normal_stimuli.draw()
                iframe += 1
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.normal_left_stimuli.draw()
                VSObject.normal_right_stimuli.draw()
                win.flip()

                samples, timestamp = inlet.pull_sample()
                predict_id = int(samples[0]) - 1  # online predict id

                if predict_id == 0:
                    response_stimuli = [VSObject.response_left_stimuli]
                    normal_stimuli = [VSObject.normal_right_stimuli]
                elif predict_id == 1:
                    response_stimuli = [VSObject.response_right_stimuli]
                    normal_stimuli = [VSObject.normal_left_stimuli]
                else:
                    response_stimuli = [
                        VSObject.response_left_stimuli,
                        VSObject.response_right_stimuli,
                    ]
                    normal_stimuli = []

                iframe = 0
                while iframe < int(fps * response_time):
                    for _response_stimuli in response_stimuli:
                        _response_stimuli.draw()
                    if normal_stimuli:
                        for _normal_stimuli in normal_stimuli:
                            _normal_stimuli.draw()
                    iframe += 1
                    win.flip()

    elif pdim == "con-ssvep":
        global online_text_pos, online_symbol_text

        if inlet:
            MyTherad = GetPlabel_MyTherad(inlet)
            MyTherad.feedbackThread()

        # config experiment settings
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            iframe += 1
            win.flip()

        # episode 2: begin to flash
        if port:
            port.setData(0)
        for trial in trials:
            # quit demo
            keys = event.getKeys(["q"])
            if "q" in keys:
                MyTherad.stop_feedbackThread()
                break

            # initialise index position
            id = int(trial["id"])
            position = VSObject.stim_pos[id] + np.array([0, VSObject.stim_width / 2])
            VSObject.index_stimuli.setPos(position)

            # phase I: speller & index (eye shifting)
            iframe = 0
            while iframe < int(fps * index_time):
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                VSObject.index_stimuli.draw()
                iframe += 1
                win.flip()

            # phase II: rest state
            if rest_time != 0:
                iframe = 0
                while iframe < int(fps * rest_time):
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.text = online_symbol_text
                        VSObject.text_response.pos = online_text_pos
                        VSObject.text_response.draw()
                    for text_stimulus in VSObject.text_stimuli:
                        text_stimulus.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                if online:
                    VSObject.rect_response.draw()
                    VSObject.text_response.text = online_symbol_text
                    VSObject.text_response.pos = online_text_pos
                    VSObject.text_response.draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

        if inlet:
            MyTherad.stop_feedbackThread()
