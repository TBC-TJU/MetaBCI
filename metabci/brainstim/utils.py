import serial
from psychopy import parallel
import numpy as np

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
        else:
            self.port = parallel.ParallelPort(address=port_addr)
    
    def setData(self, label):
        if self.use_serial:
            self.port.write(int(label))
        else:
            self.port.setData(int(label))

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
