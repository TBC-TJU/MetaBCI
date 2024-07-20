from metabci.brainflow.amplifiers import LSLInlet, DataInlet, MarkerInlet
from typing import List
import pylsl
import numpy as np


class LSLapps():
    """An amplifier implementation for Lab streaming layer (LSL) apps.
    LSL ref as: https://github.com/sccn/labstreaminglayer
    The LSL provides many builded apps for communiacting with varities
    of devices, and some of the are EEG acquiring device, like EGI, g.tec,
    DSI and so on. For metabci, here we just provide a pathway for reading 
    lsl data streams, which means as long as the the LSL providing the app, 
    the metabci could support its online application. Considering the 
    differences among different devices for tranfering the event trigger. 
    YOU MUST BE VERY CAREFUL to determine wether the data stream reading 
    from the LSL apps contains a event channel. For example, the neuroscan
    synamp II will append a extra event channel to the raw data channel.
    Because we do not have chance to test each device that LSL supported, so 
    plese modifiy this class before using with your own condition.
    """

    def __init__(self, ):
        streams = pylsl.resolve_streams()
        self.marker_inlet = None
        self.data_inlet = None
        self.device_data = None
        self.marker_data = None
        self.marker_cache = np.zeros((5, 2))
        self.marker_count = 0
        for info in streams:
            if info.type() == 'Markers':
                if info.nominal_srate() != pylsl.IRREGULAR_RATE \
                        or info.channel_format() != pylsl.cf_string:
                    print('Invalid marker stream ' + info.name())
                print('Adding marker inlet: ' + info.name())
                self.marker_inlet = MarkerInlet(info)
            elif info.nominal_srate() != pylsl.IRREGULAR_RATE \
                    and info.channel_format() != pylsl.cf_string:
                print('Adding data inlet: ' + info.name())
                self.data_inlet = DataInlet(info)
            else:
                print('Don\'t know what to do with stream ' + info.name())
   
    def save_marker(self):
        n_marker = self.marker_data.shape[0]
        self.marker_cache[self.marker_count:self.marker_count+n_marker]
        self.marker_count += n_marker

    def recv(self):
        self.device_data = self.data_inlet.stream_action()
        self.marker_data = self.marker_inlet.stream_action()
        if any(self.marker_data):
            self.save_marker()
        if any(self.device_data) and any(self.marker_cache):
            self.save_marker()
            label_column = np.zeros((self.device_data.shape[0], 1))
            insert_keys = self.device_data[:, -1].searchsorted(
                self.marker_cache[:, -1])
            label_column[insert_keys] = self.marker_cache[:, 1]
            self.device_data[:, -1] = label_column
            self.marker_cache = np.zeros((5, 2))
            self.marker_count = 0
            return self.device_data.tolist()
        elif any(self.device_data):
            self.device_data[:, -1] = 0
            return self.device_data.tolist()
        elif any(self.marker_data):
            self.save_marker()
            return []
        else:
            return []
