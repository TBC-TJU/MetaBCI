import serial
from psychopy import parallel
import numpy as np

from pylsl import StreamInfo, StreamOutlet


class NeuroScanPort:
    """Send tag communication.
    -author: Lichao Xu
    -Created on: 2020-07-30
    -update log:
        None
    Parameters
    ----------
        port_addr: ndarray,
            The port address.
        use_serial: bool,
            Send tags using serial port.
        baudrate: int,
            The serial port baud rate.
    """

    def __init__(self, port_addr, use_serial=False, baudrate=115200):
        self.use_serial = use_serial
        if use_serial:
            self.port = serial.Serial(port=port_addr, baudrate=baudrate)
            self.port.write([0])
        else:
            self.port = parallel.ParallelPort(address=port_addr)

    def setData(self, label):
        if self.use_serial:
            self.port.write([int(label)])
        else:
            self.port.setData(int(label))


class NeuraclePort:
    """Send trigger to Neuracle device
    -author: Jie Mei
    -create on: 2022-12-05
    -update log:
        None

    The Neuracle device uses serial port for writing trigger, so it
    does not need to write a 0 trigger before a int trigger. This
    class is writen under the Trigger box instruction.

    Parameters
    ----------
        port_addr: ndarray,
            The port address.
        baudrate: int,
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
    """ Creating a lab streaming layer marker, which could align with the
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


def _check_array_like(value, length=None):
    """Check array dimensions.
    -author: Lichao Xu
    -Created on: 2020-07-30
    -update log:
        None
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
    """Clear dictionary.
    -author: Lichao Xu
    -Created on: 2020-07-30
    -update log:
        None
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
