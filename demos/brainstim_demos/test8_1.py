from pylsl import StreamInfo, StreamOutlet
import time
import numpy as np

lsl_source_id = 'test'

p_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

info = StreamInfo(
            name='event_transmitter',
            type='event',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id=lsl_source_id)
outlet = StreamOutlet(info)
print('Waiting connection Amplifier...')
while True:
    if outlet.wait_for_consumers(1e-3):
        # if self.outlet.have_consumers():
        # #不停寻找同lsl_source_id的刺激程序
        break
print('Connected to Amplifier, event ready to send')
while True:
    if outlet.have_consumers():
        label = [p_labels[0]]
        outlet.push_sample(label)
        del p_labels[0]
        if len(p_labels) == 0:
            while True:
                time.sleep(2)
                print("stop sending")
        time.sleep(0.2)