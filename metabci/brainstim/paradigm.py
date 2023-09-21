# -*- coding: utf-8 -*-
import math

# load in basic modules
import os
import os.path as op
import string
import numpy as np
from math import pi
from psychopy import data, visual, event
from psychopy.visual.circle import Circle
from pylsl.pylsl import StreamInlet, resolve_byprop
from .utils import NeuroScanPort, NeuraclePort, _check_array_like
import threading
from copy import copy
import random
from scipy import signal
from PIL import Image


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


def wave_new(stim_num, type):
    """determine the color of each offset dot according to "type".
    -author: Jieyu Wu
    -Created on: 2022-12-14
    -update log:
    Parameters
    ----------
        stim_num: int,
            Number of stimuli dots of each target.
        type: int,
            avep code.

    Returns:
    ----------
        point: ndarray,
            (stim_num, 3)
    """
    point = [[-1, -1, -1] for i in range(stim_num)]
    if type == 0:
        pass
    else:
        point[type - 1] = [1, 1, 1]
    point = np.array(point)
    return point


def pix2height(win_size, pix_num):
    height_num = pix_num / win_size[1]
    return height_num


def height2pix(win_size, height_num):
    pix_num = height_num * win_size[1]
    return pix_num


def code_sequence_generate(basic_code, sequences):
    """Quickly generate coding sequences for sub-stimuli using basic endcoding units and encoding sequences.
    -author: Jieyu Wu
    -Created on: 2023-09-18
    -update log:

    Parameters
    ----------
        basic_code: list,
            Each basic encoding unit in the encoding sequence.
        sequences: list of array,
            Encoding sequences for basic_code.
    Returns:
    ----------
        code: ndarray,
            coding sequences for sub-stimuli.
    """

    code = []
    for seq_i in range(len(sequences)):
        code_list = []
        seq_length = len(sequences[seq_i])
        for code_i in range(seq_length):
            code_list.append(basic_code[sequences[seq_i][code_i]])
        code.append(code_list)
    code = np.array(code)
    return code


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
            # note that those coordinates are still not the real ones that
            # need to be set on the screen
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
        self.columns = columns
        self.rows = rows

    def config_text(
        self, unit="pix", symbols=None, symbol_height=0, tex_color=[1, 1, 1]
    ):
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
                    units=unit,
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
            height=brige_width * 3 / 3,
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
        self.reset_res_text = ">:  "
        if symbol_height == 0:
            self.symbol_height = brige_width
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

    def config_index(self, index_height=0, units="pix"):
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
            color=[1.0, -1.0, -1.0],
            colorSpace="rgb",
            units=units,
            height=index_height,
            bold=True,
            autoLog=False,
        )


# standard SSVEP paradigm


class SemiCircle(Circle):
    """
    A SemiCircle class inherited from Circle.
    """

    def _calcVertices(self):
        # only draw half of a circle
        d = np.pi / self.edges
        self.vertices = np.asarray(
            [
                np.asarray((np.sin(e * d), np.cos(e * d))) * self.radius
                for e in range(int(round(self.edges) + 1))
            ]
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
            self.stim_colors = (
                sinusoidal_sample(
                    freqs=self.freqs,
                    phases=self.phases,
                    srate=self.refresh_rate,
                    frames=self.stim_frames,
                    stim_color=stim_color,
                )
                - 1
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

    def config_color(
        self, refresh_rate=0, stim_duration=0.1, stim_ISI=0.025, stim_round=1
    ):
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
        self.stim_ISI = stim_ISI
        self.stim_round = stim_round
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
        self.stim_frames = int(
            (row_num * (stim_duration + stim_ISI) * refresh_rate)
        ) + int((col_num * (stim_duration + stim_ISI) * refresh_rate))

        # back png
        self.back = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures" + os.sep + "back.png",
        )
        self.back_stimuli = visual.ElementArrayStim(
            self.win,
            units="pix",
            elementTex=self.back,
            elementMask=None,
            texRes=2,
            nElements=1,
            sizes=[self.win_size],
            xys=np.array([[0, 0]]),
            oris=[0],
            opacities=[1],
            contrs=[-1],
        )

        # start
        self.flash_stimuli = []
        self.order_index = np.zeros([int((row_num + col_num) * self.stim_round)])
        for round_num in range(self.stim_round):
            row_order_index = list(range(0, row_num))
            np.random.shuffle(row_order_index)
            col_order_index = list(range(0, col_num))
            np.random.shuffle(col_order_index)
            # reset event label
            l_row_order_index = [x + 1 for x in row_order_index]
            l_col_order_index = [x + row_num + 1 for x in col_order_index]

            order_row_col = np.array(l_row_order_index + l_col_order_index)
            # print(order_row_col.shape)
            self.order_index[
                (round_num * (row_num + col_num)) : (
                    (round_num + 1) * (row_num + col_num)
                )
            ] = order_row_col[
                :
            ]  # event label
            # print(self.order_index)

            # Determine row and column char status
            stim_colors_row = np.zeros(
                [
                    (row_num * col_num),
                    int(row_num * refresh_rate * (stim_duration + stim_ISI)),
                    3,
                ]
            )
            stim_colors_col = np.zeros(
                [
                    (row_num * col_num),
                    int(col_num * refresh_rate * (stim_duration + stim_ISI)),
                    3,
                ]
            )  #
            row_label = np.zeros(
                [int(row_num * refresh_rate * (stim_duration + stim_ISI))]
            )
            col_label = np.zeros(
                [int(col_num * refresh_rate * (stim_duration + stim_ISI))]
            )

            tmp = 0
            for col_i in col_order_index:
                stim_colors_col[
                    (col_i * row_num) : ((col_i + 1) * row_num),
                    int(tmp * refresh_rate * (stim_duration + stim_ISI)) : int(
                        tmp * refresh_rate * (stim_duration + stim_ISI)
                        + refresh_rate * (stim_duration)
                    ),
                ] = [-1, -1, -1]
                col_label[int(tmp * refresh_rate * (stim_duration + stim_ISI))] = 1
                tmp += 1

            tmp = 0
            for row_i in row_order_index:
                for col_i in range(col_num):
                    stim_colors_row[
                        (row_i + row_num * col_i),
                        int(tmp * refresh_rate * (stim_duration + stim_ISI)) : int(
                            tmp * refresh_rate * (stim_duration + stim_ISI)
                            + refresh_rate * stim_duration
                        ),
                    ] = [-1, -1, -1]
                    row_label[int(tmp * refresh_rate * (stim_duration + stim_ISI))] = 1
                tmp += 1

            stim_colors = np.concatenate((stim_colors_row, stim_colors_col), axis=1)
            self.roworcol_label = np.concatenate(
                (row_label, col_label), axis=0
            )  # each round is the same
            self.stim_colors = np.transpose(stim_colors, [1, 0, 2])

            # add flashing targets onto interface
            for sf in range(self.stim_frames):
                self.flash_stimuli.append(
                    visual.ElementArrayStim(
                        win=self.win,
                        units="pix",
                        nElements=self.n_elements,
                        opacities=np.ones((self.n_elements,)) * 0.7,
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
            "textures" + os.sep + "left_hand.png",
        )
        self.tex_right = os.path.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))),
            "textures" + os.sep + "right_hand.png",
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


# standard AVEP paradigm


class AVEP(VisualStim):
    """Create AVEP stimuli.
    -author: Jieyu Wu
    -Created on: 2022-12-14
    -update log:
        2022-12-17 by Jieyu Wu
    Parameters
    ----------
    win:
        The window object.
    dot_shape: str
        The pattern of stimuli.
    n_rep: int
        repetitions of stimulation.
    duty: float
        PWM of a single flicker.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.
    cluster_number: int
        Number of dots in cluster of stimuli.

    """

    def __init__(
        self,
        win,
        dot_shape="circle",
        n_rep=5,
        duty=0.5,
        cluster_num=1,
        colorSpace="rgb",
        allowGUI=True,
    ):
        """Item class from VisualStim.

        Args:

        """
        self.dot_shape = dot_shape
        self.n_rep = n_rep
        if dot_shape == "cluster" and cluster_num == 1:
            self.cluster_num = 6
            dot_shape == "circle"
        elif dot_shape == "cluster":
            self.cluster_num = cluster_num
            dot_shape == "circle"
        else:
            self.cluster_num = cluster_num

        self.duty = duty
        super().__init__(win=win, colorSpace=colorSpace, allowGUI=allowGUI)

    def config_array(self, frequencies=None):
        """Config the dot array according to the code sequences.
        Parameters
        ----------
            frequencies: array,
                frequencies of each target.
        """
        stim_time = self.stim_time
        stim_frames = self.stim_frames
        n_element = self.n_elements
        sequence = self.sequence
        if frequencies is None:
            frequencies = 10 * np.ones((n_element, 1))
        t = np.linspace(0, stim_time, stim_frames, endpoint=False)
        stim_ary = [[] for i in range(n_element)]
        for target_i in range(n_element):
            tar_fre = frequencies[target_i]
            avep_num = int(tar_fre * stim_time)
            fold_num = int(np.ceil(avep_num / len(sequence[target_i])))
            tar_seq = np.tile(sequence[target_i], fold_num)[0:avep_num]
            sample = (signal.square(2 * pi * tar_fre * t, duty=self.duty) + 1) / 2
            sample = sample.astype(int)
            a = np.append(0, sample)
            b = np.diff(a)
            c = np.where(b == 1)
            c = np.append(c, sample.shape[0])
            d = np.array([], "int")
            for avep_i in range(avep_num):
                d = np.append(d, sample[c[avep_i] : c[avep_i + 1]] * tar_seq[avep_i])
            stim_ary[target_i] = d
        self.stim_ary = []
        for clu_i in range(self.cluster_num):
            self.stim_ary.extend(stim_ary)

    def config_dot_pos(self, offset=None):
        """Config the position of each single dot.
        Parameters
        ----------
            offset: list,
                offset distance between stimulus point and target center.
        """
        if offset is None and self.stim_num == 2:
            offset = [[-20, -20], [20, -20]]
        elif offset is None and self.stim_num == 4:
            offset = [[20, 20], [-20, 20], [-20, -20], [20, -20]]
        elif offset is None and self.stim_num == 8:
            offset = [
                [20, 20],
                [0, 20],
                [-20, 20],
                [-20, 0],
                [-20, -20],
                [0, -20],
                [20, -20],
                [20, 0],
            ]
        elif len(offset) == self.stim_num:
            pass
        else:
            raise Exception("Please confirm the offset position list!")
        dot_pos = np.zeros((self.n_elements, self.stim_num, 2))
        for dot_i in range(self.stim_num):
            dot_pos[:, dot_i, :] = self.stim_pos
            dot_pos[:, dot_i, 0] = dot_pos[:, dot_i, 0] + offset[dot_i][0]
            dot_pos[:, dot_i, 1] = dot_pos[:, dot_i, 1] + offset[dot_i][1]
        if self.cluster_num == 1:
            self.stim_dot_pos = np.tile(
                dot_pos[np.newaxis, ...], (self.stim_frames, 1, 1, 1)
            )
        else:
            self.stim_dot_pos = np.zeros(
                (self.stim_frames, self.cluster_num * self.n_elements, self.stim_num, 2)
            )
            for stim_i in range(self.stim_frames):
                for clu_i in range(self.cluster_num):
                    width_rand = random.randint(-3, 3)
                    height_rand = random.randint(-3, 3)
                    self.stim_dot_pos[
                        stim_i,
                        clu_i * self.n_elements : (clu_i + 1) * self.n_elements,
                        :,
                        0,
                    ] = (
                        dot_pos[..., 0] + width_rand
                    )
                    self.stim_dot_pos[
                        stim_i,
                        clu_i * self.n_elements : (clu_i + 1) * self.n_elements,
                        :,
                        1,
                    ] = (
                        dot_pos[..., 1] + height_rand
                    )

    def config_dot_color(self):
        """Config color array according to dot array."""
        stim_num = self.stim_num
        stim_ary = self.stim_ary
        stim_colors = np.zeros(
            (self.stim_frames, self.n_elements * self.cluster_num, stim_num, 3)
        )
        for tar_i in range(self.n_elements * self.cluster_num):
            for frame_i in range(self.stim_frames):
                dot_type = stim_ary[tar_i][frame_i]
                stim_colors[frame_i, tar_i, :, :] = wave_new(
                    stim_num=stim_num, type=dot_type
                )
        self.stim_colors = stim_colors

    def config_color(
        self, refresh_rate, stim_time, stim_color, sequence, stim_opacities=1, **kwargs
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
            sequence: list,
                Spatial codes of each stimulus
            stim_opacities: int or float,
                Opacities of each stimulus.
            freqs: list of float,
                Frequencies of each stimulus.
            stim_num: int
                Numeber of stimuli dots of each target.

        Raises:
            Exception: Inconsistent frames and color matrices.
        """

        # initialize extra inputs
        all_shapes = [
            "sin",
            "sqr",
            "saw",
            "tri",
            "sinXsin",
            "sqrXsqr",
            "circle",
            "gauss",
            "cross",
        ]
        self.refresh_rate = refresh_rate
        self.stim_time = stim_time
        self.stim_color = stim_color
        self.stim_opacities = stim_opacities
        self.stim_frames = int(stim_time * self.refresh_rate)
        self.sequence = sequence

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
        if "stim_num" in kwargs.keys():
            self.stim_num = kwargs["stim_num"]
        # config the location of each dot
        self.config_dot_pos()
        # create the coding array according to the input spatial codes
        self.config_array(frequencies=self.freqs)
        # generate the color array according to coding array of each stimulus
        self.config_dot_color()
        # the dot number equal to number of targets multiplied by the number of stimuli of avep
        # and the number of cluster dots
        all_dot_num = self.n_elements * self.stim_num * self.cluster_num
        stim_colors = np.concatenate(
            [self.stim_colors[:, :, i, :] for i in range(self.stim_num)], axis=1
        )
        stim_dot_pos = np.concatenate(
            [self.stim_dot_pos[:, :, i, :] for i in range(self.stim_num)], axis=1
        )
        stim_size = np.concatenate(
            [self.stim_sizes for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        stim_oris = np.concatenate(
            [self.stim_oris for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        stim_sfs = np.concatenate(
            [self.stim_sfs for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        stim_contrs = np.concatenate(
            [self.stim_contrs for i in range(self.stim_num * self.cluster_num)], axis=0
        )
        if self.dot_shape == "cluster":
            dot_shape = "circle"
        elif self.dot_shape == "square":
            dot_shape = None
        elif self.dot_shape in all_shapes:
            dot_shape = self.dot_shape
        else:
            raise Exception("Please input the correct shape!")

        incorrect_frame = stim_colors.shape[0] != self.stim_frames
        incorrect_number = stim_colors.shape[1] != all_dot_num
        if incorrect_frame or incorrect_number:
            raise Exception("Incorrect color matrix or flash frames!")
        # add flashing targets onto interface
        self.flash_stimuli = []
        for sf in range(self.stim_frames):
            self.flash_stimuli.append(
                visual.ElementArrayStim(
                    win=self.win,
                    units="pix",
                    nElements=all_dot_num,
                    sizes=stim_size,
                    xys=stim_dot_pos[sf, ...],
                    colors=stim_colors[sf, ...],
                    opacities=self.stim_opacities,
                    oris=stim_oris,
                    sfs=stim_sfs,
                    contrs=stim_contrs,
                    elementTex=np.ones((64, 64)),
                    elementMask=dot_shape,
                    texRes=48,
                )
            )

    def num2bin_ary(self, num, n_elements, type="0-1"):
        """Converts a decimal number to a binary sequence of specified bit.
        The byte-codes of the binary sequence are 1 and 2
        -author: Jieyu Wu
        -Created on: 2022-12-16
        -update log:
            2023-3-27 by Shihang Yu
        Parameters
        ----------
            num : int,
                A decimal number.
            n_elements : int
                Num of stimulus.
            type : int,
                if type is '0-1',convert each '0' to '1-2' and each '1' to '2-1'.
                else convert each '0' to '1' and each '1' to '2'.

        Returns:
        ----------
            bin_ary2: list,
                (stim_num, 3)
        """
        bit = int(math.ceil(math.log(n_elements) / math.log(2)))
        bin_ary = np.zeros(bit, "int")
        quo = num
        i = -1
        while quo != 0:
            (quo, mod) = divmod(quo, 2)
            bin_ary[i] = mod
            i -= 1
        if type == "0-1":
            bin_ary2 = np.zeros(bit * 2, "int")
            elements = np.array([[0, 1], [1, 0]])
            for j in range(bit):
                bin_ary2[j * 2 : (j + 1) * 2] = elements[int(bin_ary[j])]
        else:
            bin_ary2 = bin_ary

        return list(bin_ary2 + 1)


# standard SSaVEP paradigm


class SSAVEP(VisualStim):
    """Create SSAVEP stimuli.
    -author: Jieyu Wu
    -Created on: 2023-09-11
    -update log:

    Parameters
    ----------
    win:
        The window object.
    n_elements: int
        The number of unique stimuli.
    n_members: int
        The number of sub-stimuli in an individual command.
    colorspace: str,
        The color space, default to rgb.
    allowGUI: bool
        Defaults to True, which allows frame-by-frame drawing and key-exit.

    """

    def __init__(
        self, win, n_elements=20, n_members=8, colorSpace="rgb", allowGUI=True
    ):
        self.n_members = n_members
        self.n_elements = n_elements
        self.n_groups = self.n_members * self.n_elements
        super().__init__(win, colorSpace, allowGUI)

    def config_member_pos(
        self,
        win,
        radius=0.1,
        angles=[0],
        outter_deg=4,
        inner_deg=1.5,
        tex_pix=128,
        sep_line_pix=16,
    ):
        """Config color of stimuli.
        Parameters
        ----------
            win: psychopy.visual.Window,
                window this shape is being drawn to.
            radius: float,
                related to sizes of stimulus.
            angles: list of float,
                initial rotation angle of each sub-stimulus.
            outter_deg: float,
                the ratio determining the size of two circles, used in self.generate_octants.
            inner_deg: float,
                the ratio determining the size of two circles, used in self.generate_octants.
            tex_pix: int
                size of sub-stimulus, used in self.generate_octants
        """
        win_size = np.array(win.size)
        lpad, rpad = int(0.15 * win_size[1]), int(0.15 * win_size[1])
        upad, dpad = int(0.1 * win_size[1]), int(0.1 * win_size[1])
        x_stride = (win_size[0] - lpad - rpad) // (self.columns - 1)
        y_stride = (win_size[1] - upad - dpad) // (self.rows - 1)
        positions = np.array(
            [
                (
                    (lpad + (i % self.columns) * x_stride - win_size[0] / 2)
                    / win_size[1],
                    (win_size[1] / 2 - upad - (i // self.columns) * y_stride)
                    / win_size[1],
                )
                for i in range(self.n_elements)
            ]
        )
        self.positions_rad = positions
        self.stim_pos = self.positions_rad
        self.element_mask = np.zeros((self.n_groups, self.n_members), dtype=np.bool)
        self.radius = radius
        self.angles = angles
        self.angles = np.tile(np.reshape(self.angles, (-1, 1)), (1, self.n_members))
        self.stim_pos = np.array(self.stim_pos)
        self.positions = np.tile(positions, (1, self.n_members))
        self.member_angles = np.append(
            360 / self.n_members * np.arange(1, self.n_members), 0
        )
        # 计算每个圆环的位置
        oris = self.angles + self.member_angles.reshape((1, -1))
        oris = np.reshape(oris, (-1))
        self.oris = oris
        thetas = np.radians(oris)
        rotate_mat = np.array(
            [[np.cos(thetas), np.sin(thetas)], [-np.sin(thetas), np.cos(thetas)]]
        )
        self.member_positions = np.tensordot(
            rotate_mat, np.array([-0.5, 0.5]) * self.radius, axes=((1), (0))
        ).T
        self.member_positions = np.reshape(self.member_positions, (self.n_elements, -1))
        xys = self.positions + self.member_positions
        xys = np.reshape(xys, (-1, 2))
        self.element_pos = xys
        self.outter_deg = outter_deg
        self.inner_deg = inner_deg
        self.tex_pix = tex_pix
        self.sep_line_pix = sep_line_pix

    def config_stim(
        self,
        win,
        sizes=[[0.1, 0.1]],
        stim_color=[[1.0, 1.0, 1.0]],
        stim_opacities=[1],
        member_degree=None,
    ):
        """Config color of stimuli.
        Parameters
        ----------
            win: psychopy.visual.Window,
                window this shape is being drawn to.
            sizes: list of float,
                sizes of the state stimuli.
            stim_color: list of float,
                rgb value of the state stimuli.
            stim_opacities: list of float,
                opacities of the state stimuli.
            member_degree: list of float
                rotation angle of each sub-stimulus
        """
        self.generate_octants(
            win,
            outter_deg=self.outter_deg,
            inner_deg=self.inner_deg,
            member_degree=member_degree,
        )
        self.elementTex = "adaptive_octants.png"
        self.stim_oris = np.zeros((self.n_elements,))  # orientation
        self.stim_sfs = np.zeros((self.n_elements,))  # spatial frequency
        self.stim_contrs = [1]  # contrast
        self.stim_opacities = stim_opacities
        colors = np.tile(stim_color, (1, self.n_groups, 1))

        self.state_stim = self.create_elements(
            win,
            units="height",
            elementTex=self.elementTex,
            elementMask=None,
            nElements=self.n_groups,
            frames=1,
            sizes=sizes,
            xys=self.element_pos,
            oris=self.oris,
            colors=colors,
            opacities=[1],
            contrs=[1],
            texRes=2,
        )

    def config_ring(
        self, win, sizes=[[0.3, 0.3]], ring_colors=[1, 1, 1], opacities=[1.0]
    ):
        _TEX = op.join(
            op.abspath(op.dirname(op.abspath(__file__))), "textures", "ring.png"
        )
        """Config color of rings around the stimuli.
        Parameters
        ----------
            win: psychopy.visual.Window,
                window this shape is being drawn to.
            sizes: list of float,
                sizes of each ring.
            ring_colors: list of float,
                rgb value of rings.
            opacities: list of float,
                opacities of center target.
        """
        sizes = sizes
        ring_colors1 = np.tile(ring_colors, (1, self.n_elements, 1))
        self.ring = self.create_elements(
            win,
            units="height",
            elementTex=_TEX,
            elementMask=None,
            nElements=self.n_elements,
            frames=1,
            sizes=sizes,
            xys=self.positions_rad,
            colors=ring_colors1,
            opacities=opacities,
            contrs=[1],
            texRes=2,
        )

    def config_target(
        self, win, sizes=[[0.2, 0.2]], target_colors=[1, 0, 0], opacities=[1.0]
    ):
        """Config color of targets at the center of each stimulus.
        Parameters
        ----------
            win: psychopy.visual.Window,
                window this shape is being drawn to.
            sizes: list of float,
                sizes of each center target.
            target_colors: list of float,
                rgb value of center targets.
            opacities: list of float,
                opacities of center targets.
        """
        _TEX = op.join(
            op.abspath(op.dirname(op.abspath(__file__))), "textures", "centroid.png"
        )
        target_colors1 = np.tile(target_colors, (1, self.n_elements, 1))
        self.center_target = self.create_elements(
            win,
            units="height",
            elementTex=_TEX,
            elementMask=None,
            nElements=self.n_elements,
            frames=1,
            sizes=sizes,
            xys=self.positions_rad,
            oris=[45],
            colors=target_colors1,
            opacities=opacities,
            contrs=[1],
            texRes=2,
        )

    def config_flash_array(
        self,
        refresh_rate=60,
        freqs=[15],
        phases=[0],
        codes=[[0], [1], [2], [3]],
        stim_time_member=0.5,
        stim_color=[1, 1, 1],
        stimtype="sinusoid",
    ):
        """Config flash sequence array of stimuli.
        Parameters
        ----------
            refresh_rate: int or float,
                Refresh rate of screen.
            freqs: list of float,
                Frequencies of each stimulus.
            phases: list of float,
                Phases of each stimulus.
            codes: list of list,
                code sequences of each stimulus.
            stim_time_member: float,
                Time of each sub-stimulus.
            stim_color: list of float,
                Maximum rgb value during stimuli flicker.
            stimtype: str,
                flashing mode of stimuli .
        """
        self.codes = np.array(codes)
        self.freqs = np.array(freqs)
        self.freqs = np.reshape(
            np.tile(np.reshape(self.freqs, (-1, 1)), (1, self.n_members)),
            (self.freqs.size * self.n_members, 1),
        )
        self.phases = np.array(phases)
        self.phases = np.reshape(
            np.tile(np.reshape(self.phases, (-1, 1)), (1, self.n_members)),
            (self.phases.size * self.n_members, 1),
        )
        self.refresh_rate = refresh_rate
        self.stim_frames_member = int(stim_time_member * self.refresh_rate)
        self.stim_time_member = stim_time_member
        self.stimtype = stimtype
        if stimtype == "sinusoid":
            self.stim_colors_member = (
                sinusoidal_sample(
                    freqs=self.freqs,
                    phases=self.phases,
                    srate=self.refresh_rate,
                    frames=self.stim_frames_member,
                    stim_color=stim_color,
                )
                - 1
            )
        self.n_sequence = np.shape(self.codes)[1]
        self.stim_time = self.stim_time_member * self.n_sequence
        self.stim_frames = self.stim_frames_member * self.n_sequence
        self.stim_colors1 = np.ones(
            (self.stim_frames_member, self.n_groups, self.n_sequence, 3)
        )
        for tar_idx in range(self.n_elements):
            tar_codes = self.codes[tar_idx]
            for seq_idx in range(self.n_sequence):
                for seq_group_idx in range(len(tar_codes[seq_idx])):
                    self.stim_colors1[
                        :,
                        tar_idx * self.n_members + tar_codes[seq_idx][seq_group_idx],
                        seq_idx,
                        :,
                    ] = self.stim_colors_member[
                        :,
                        tar_idx * self.n_members + tar_codes[seq_idx][seq_group_idx],
                        :,
                    ]
        self.stim_colors = np.concatenate(
            [self.stim_colors1[:, :, i, :] for i in range(self.n_sequence)], axis=0
        )

    def config_color(
        self,
        win,
        refresh_rate=60,
        freqs=[15],
        phases=[0],
        codes=[[0], [1], [2], [3]],
        stim_time_member=0.5,
        stim_color=[1.0, 1.0, 1.0],
        stimtype="sinusoid",
        sizes=[0.1, 0.1],
    ):
        """Config color of stimuli.
        Parameters
        ----------
            refresh_rate: int or float,
                Refresh rate of screen.
            freqs: list of float,
                Frequencies of each stimulus.
            phases: list of float,
                Phases of each stimulus.
            codes: list of list,
                code sequences of each stimulus.
            stim_time_member: float,
                Time of each sub-stimulus.
            stim_color: list of float,
                Maximum rgb value during stimuli flicker.
            stimtype: str,
                flashing mode of stimuli .
            sizes: list of float,
                sizes of each sub-stimulus.
        """
        self.config_flash_array(
            refresh_rate, freqs, phases, codes, stim_time_member, stim_color, stimtype
        )
        self.flash_stimuli = self.create_elements(
            win,
            units="height",
            elementTex=self.elementTex,
            elementMask=None,
            nElements=self.n_groups,
            frames=self.stim_frames,
            sizes=sizes,
            xys=self.element_pos,
            oris=self.oris,
            colors=self.stim_colors,
            opacities=self.stim_opacities,
            contrs=self.stim_contrs,
            texRes=2,
        )

    def generate_octants(self, win, outter_deg=4, inner_deg=2, member_degree=None):
        """Generate the sub-stimulus and save the .png file.
        Parameters
        ----------
            win: psychopy.visual.Window,
                window this shape is being drawn to.
            radius: float,
                related to sizes of stimulus.
            angles: list of float,
                initial rotation angle of each sub-stimulus.
            outter_deg: float,
                the ratio determining the size of two circles, used in self.generate_octants.
            inner_deg: float,
                the ratio determining the size of two circles, used in self.generate_octants.
            member_degree: list of float
                rotation angle of differen sub-stimuli in a single stimulus.
        """
        if member_degree is None:
            member_degree = self.member_angles[0]
        win_size = win.size
        win_size = [1600, 900]
        win.color = [-1, -1, -1]
        radius = self.tex_pix / win_size[1]
        sep_line_height = self.sep_line_pix / win_size[1]
        ratio = np.tan(np.radians(inner_deg)) / np.tan(np.radians(outter_deg))
        edges = 256
        outter_circle = visual.Circle(
            win,
            units="height",
            radius=radius,
            edges=edges,
            fillColor="white",
            lineColor="white",
        )
        inner_circle = visual.Circle(
            win,
            units="height",
            radius=radius * ratio,
            edges=edges,
            fillColor="black",
            lineColor="black",
        )
        h_line = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=0,
        )
        v_line = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=90,
        )
        diag_line1 = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=90 - member_degree + 0.001,
        )
        diag_line2 = visual.Rect(
            win,
            units="height",
            size=(win_size[0] / win_size[1], sep_line_height),
            fillColor="black",
            lineColor=None,
            ori=-90 + member_degree - 0.001,
        )
        semi_circle1 = SemiCircle(
            win,
            units="height",
            radius=radius * 1.2,
            edges=edges,
            fillColor="black",
            lineColor="black",
            ori=180 - member_degree,
        )
        semi_circle2 = SemiCircle(
            win,
            units="height",
            radius=radius * 1.2,
            edges=edges,
            fillColor="black",
            lineColor="black",
            ori=0,
        )
        stims = [
            outter_circle,
            inner_circle,
            h_line,
            v_line,
            diag_line1,
            diag_line2,
            semi_circle1,
            semi_circle2,
        ]
        # stims = [
        #     outter_circle, inner_circle
        # ]
        rect = [-2 * radius * win_size[1] / win_size[0], 0, 0, -2 * radius]
        screenshot = visual.BufferImageStim(win, stim=stims, buffer="back", rect=rect)
        image = Image.fromarray(np.array(screenshot.image), mode="RGB")
        image.putalpha(image.convert("L"))
        image.save("adaptive_octants.png")
        win.clearBuffer()

    def create_elements(
        self,
        win,
        units="pix",
        elementTex=None,
        elementMask=None,
        nElements=1,
        frames=1,
        sizes=[[0.1, 0.1]],
        xys=[[0, 0]],
        oris=[0],
        colors=[[1, 1, 1]],
        contrs=[1],
        opacities=[1],
        texRes=48,
    ):
        """create the specific elements.
        Parameters
        ----------
            win: psychopy.visual.Window,
                window this shape is being drawn to.
            units: str,
                units to use when drawing.
            elementTex: object,
                texture data of the elements.
            nElements: int,
                number of the elements.
            frames: int,
                number of frames drawn.
            sizes: list of float
                size of the elements.
            xys: list of float
                position of the elements.
            oris: list of float
                rotation angle of the elements.
            colors: list of float
                colors of the elements.
            contrs: list of float
                contrast ratio of the elements.
            opacities: list of float
                opacities ratio of the elements.
            texRes: int
                the resolution of the texture
        """
        sizes = (
            np.repeat(sizes, nElements, axis=0) if len(sizes) == 1 else np.array(sizes)
        )
        xys = np.repeat(xys, nElements, axis=0) if len(xys) == 1 else np.array(xys)
        oris = np.repeat(oris, nElements, axis=0) if len(oris) == 1 else np.array(oris)

        contrs = (
            np.repeat(contrs, nElements, axis=0)
            if len(contrs) == 1
            else np.array(contrs)
        )
        opacities = (
            np.repeat(opacities, nElements, axis=0)
            if len(opacities) == 1
            else np.array(opacities)
        )
        stim = []
        for frame_i in range(frames):
            stim.append(
                visual.ElementArrayStim(
                    win,
                    units=units,
                    elementTex=elementTex,
                    elementMask=elementMask,
                    texRes=texRes,
                    nElements=nElements,
                    sizes=sizes,
                    xys=xys,
                    oris=oris,
                    colors=colors[frame_i, ...],
                    opacities=opacities,
                    contrs=contrs,
                )
            )

        return stim


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
    device_type="NeuroScan",
):
    """Passing outsied parameters to inner attributes.
    -author: Wei Zhao
    -Created on: 2022-07-30
    -update log:
        2022-08-10 by Wei Zhao
        2022-08-03 by Shengfu Wen
        2022-12-05 by Jie Mei
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
        device_type: str,
            See support device list in brainstim README file
    """

    if not _check_array_like(bg_color, 3):
        raise ValueError("bg_color should be 3 elements array-like object.")
    win.color = bg_color
    fps = VSObject.refresh_rate

    if device_type == "NeuroScan":
        port = NeuroScanPort(port_addr, use_serial=True) if port_addr else None
    elif device_type == "Neuracle":
        port = NeuraclePort(port_addr) if port_addr else None
    else:
        raise KeyError(
            "Unknown device type: {}, please check your input".format(device_type)
        )
    port_frame = int(0.05 * fps)

    inlet = False
    if online:
        if (
            pdim == "ssvep"
            or pdim == "p300"
            or pdim == "con-ssvep"
            or pdim == "avep"
            or pdim == "ssavep"
        ):
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
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
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

    elif pdim == "avep":
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
            position = VSObject.stim_pos[id] + np.array([0, VSObject.tex_height / 2])
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

            for round_i in range(VSObject.n_rep):
                # phase II: rest state
                if rest_time != 0:
                    iframe = 0
                    while iframe < int(fps * rest_time):
                        if iframe == 0 and port and online and round_i == 0:
                            VSObject.win.callOnFlip(port.setData, 2)
                        elif iframe == 0 and port and round_i == 0:
                            VSObject.win.callOnFlip(port.setData, id + 1)
                            print(id + 1)
                        if iframe == port_frame and port and round_i == 0:
                            port.setData(0)

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
                        VSObject.win.callOnFlip(port.setData, 2)
                    elif sf == 0 and port:
                        VSObject.win.callOnFlip(port.setData, round_i + 1)
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
            VSObject.back_stimuli.draw()
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
                    VSObject.win.callOnFlip(port.setData, id + 21)
                elif iframe == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 21)
                if iframe == port_frame and port:
                    port.setData(0)
                VSObject.back_stimuli.draw()
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
                    VSObject.back_stimuli.draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating

            tmp = 0
            nonzeros_label = 0
            for round_num in range(VSObject.stim_round):
                for sf in range(VSObject.stim_frames):
                    if VSObject.roworcol_label[sf] > 0 and port:
                        VSObject.win.callOnFlip(
                            port.setData,
                            VSObject.order_index[tmp],
                        )
                        nonzeros_label = sf
                        tmp += 1

                        # time_recrod.append(time.time())
                        # T = time_recrod[-1] - time_recrod[-2]
                        # print('P3:%s毫秒' % ((T)*1000))
                    if (sf - nonzeros_label) > port_frame and port:
                        port.setData(0)

                    # for text_stimulus in VSObject.text_stimuli:
                    #     text_stimulus.draw()
                    VSObject.back_stimuli.draw()
                    VSObject.flash_stimuli[
                        round(round_num * VSObject.stim_frames + sf)
                    ].draw()
                    if online:
                        VSObject.rect_response.draw()
                        VSObject.text_response.draw()
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
            # {"id": 2, "name": "both_hands"},
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
                    VSObject.win.callOnFlip(port.setData, id + 1)
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

    elif pdim == "ssavep":
        conditions = [{"id": i} for i in range(VSObject.n_elements)]
        trials = data.TrialHandler(conditions, nrep, name="experiment", method="random")

        # start routine
        # episode 1: display speller interface
        iframe = 0
        while iframe < int(fps * display_time):
            VSObject.ring[0].draw()
            VSObject.state_stim[0].draw()
            if online:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
            for text_stimulus in VSObject.text_stimuli:
                text_stimulus.draw()
            VSObject.ring[0].draw()
            VSObject.state_stim[0].draw()
            # VSObject.center_target[0].draw()
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
                VSObject.ring[0].draw()
                VSObject.state_stim[0].draw()
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
                    VSObject.ring[0].draw()
                    VSObject.state_stim[0].draw()
                    iframe += 1
                    win.flip()

            # phase III: target stimulating
            for sf in range(VSObject.stim_frames):
                if sf == 0 and port and online:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                elif sf == 0 and port:
                    VSObject.win.callOnFlip(port.setData, id + 1)
                if sf == port_frame and port:
                    port.setData(0)
                VSObject.flash_stimuli[sf].draw()
                VSObject.ring[0].draw()
                for text_stimulus in VSObject.text_stimuli:
                    text_stimulus.draw()
                win.flip()

            # phase IV: respond
            if inlet:
                VSObject.rect_response.draw()
                VSObject.text_response.draw()
                VSObject.ring[0].draw()
                VSObject.state_stim[0].draw()
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
