import serial
from psychopy import parallel
import numpy as np
from demos.brainstim_demos.app_test import APP
from pylsl import StreamInfo, StreamOutlet
from multiprocessing import Event
import pyglet
import time


class NeuroScanPort:
    """
    Send tag communication Using parallel port or serial port.

    author: Lichao Xu

    Created on: 2020-07-30

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.

    Attributes
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.
        port:
            Send tag communication Using parallel port or serial port.

    Tip
    ----
    .. code-block:: python
       :caption: An example of using port to send tags

        from brainstim.utils import NeuroScanPort
        port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
        VSObject.win.callOnFlip(port.setData, 1)
        port.setData(0)

    """

    def __init__(self, port_addr, use_serial=False, baudrate=115200):
        self.use_serial = use_serial
        if use_serial:
            self.port = serial.Serial(port=port_addr, baudrate=baudrate)
            self.port.write([0])
        else:
            self.port = parallel.ParallelPort(address=port_addr)

    def setData(self, label):
        """Send event labels

        Parameters
        ----------
            label:
                The label sent.

        """
        if self.use_serial:
            self.port.write([int(label)])
        else:
            self.port.setData(int(label))


class NeuraclePort:
    """
    Send trigger to Neuracle device.The Neuracle device uses serial
    port for writing trigger, so it does not need to write a 0 trigger
    before a int trigger. This class is writen under the Trigger box instruction.

    author: Jie Mei

    Created on: 2022-12-05

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        baudrate: int
            The serial port baud rate.

    """

    def __init__(self, port_addr, baudrate=115200) -> None:
        # The only choice for neuracle is using serial for writting trigger
        self.port = serial.Serial(port=port_addr, baudrate=baudrate)

    def setData(self, label):
        # Neuracle doesn't need 0 trigger before a int trigger.
        if str(label) != '0':
            head_string = '01E10100'
            hex_label = str(hex(label))
            if len(hex_label) == 3:
                hex_value = hex_label[2]
                hex_label = '0'+hex_value.upper()
            else:
                hex_label = hex_label[2:].upper()
            send_string = head_string+hex_label
            send_string_byte = [int(send_string[i:i+2], 16)
                                for i in range(0, len(send_string), 2)]
            self.port.write(send_string_byte)


class LsLPort:
    """
    Creating a lab streaming layer marker, which could align with the
    stream which retriving stream from devices.

    """

    def __init__(self) -> None:
        self.info = StreamInfo(
            name='LSLMarkerStream',
            type='Marker',
            channel_count=1,
            nominal_srate=0,
            channel_format='cf_int16')
        self.outlet = StreamOutlet(self.info)

    def setData(self, label):
        # We don't need 0 trigger before a int trigger
        if str(label) != '0':
            self.outlet.push_sample(str(label))



class Light_trigger(APP):
    def __init__(self, lsl_source_id="trigger", w=1920, h=1080, screen_id=0):
        self.trigger_ = Event()
        self._exit = Event()
        self._exit.clear()
        self.trigger_.clear()
        self.outlet = []
        self.start_setData = False
        self.lsl_source_id = lsl_source_id

        super().__init__(name='stim_pos_setting', w=w, h=h, screen_id=screen_id)

    def control(self, dt):
        super().control(dt)
        if self._exit.is_set():
            pyglet.app.exit()

    def main_App(self):
        self.get_win(win_style='overlay')
        self.button_setting()
        self.reg_handlers(self.on_draw)
        pyglet.app.run()

    def button_setting(self):
        #self.sq = pyglet.shapes.Rectangle(x=self.w-200, y=0, width=200, height=200)
        self.sq = pyglet.shapes.Rectangle(x=0, y=0, width=self.w, height=int(self.h / 8))
        self.sq.color = (0, 0, 0)

    def on_draw(self):
        self.window.clear()
        if self.trigger_.is_set():
            self.sq.color = (250, 250, 250)
            self.sq.draw()
            self.trigger_.clear()
            self.sq.color = (0, 0, 0)

        else:
            self.sq.draw()

    def setData(self, event):
        if event == -1:
            self._exit.set()
        elif self.start_setData and event != 0:
            self.trigger_.set()
            while not self.outlet.have_consumers():
                time.sleep(0.1)
            self.outlet.push_sample([event])
            print("send event succeed", event)
        if not self.start_setData:
            print("--------------------------port start--------------------------")
            info = StreamInfo(
                name='event_transmitter',
                type='event',
                channel_count=1,
                nominal_srate=0,
                channel_format='int32',
                source_id=self.lsl_source_id)
            self.outlet = StreamOutlet(info)
            print('Waiting for Amplifier...')
            self.start_setData = True




def _check_array_like(value, length=None):
    """
    Check array dimensions.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        value: ndarray,
            The array to check.
        length: int,
            The array dimension.

    """

    flag = isinstance(value, (list, tuple, np.ndarray))
    return flag and (len(value) == length if length is not None else True)


def _clean_dict(old_dict, includes=[]):
    """
    Clear dictionary.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        old_dict: dict,
            The dict to clear.
        includes: list,
            Key-value indexes that need to be preserved.

    """

    names = list(old_dict.keys())
    for name in names:
        if name not in includes:
            old_dict[name] = None
            del old_dict[name]
    return old_dict
