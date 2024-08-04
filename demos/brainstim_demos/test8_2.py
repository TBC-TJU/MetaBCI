from pylsl.pylsl import StreamInlet, resolve_byprop
import time

lsl_source_id = 'test'

if lsl_source_id:
    inlet = True
    streams = resolve_byprop(
        "source_id", lsl_source_id, timeout=5
    )  # Resolve all streams by source_id
    if streams:
        inlet = StreamInlet(streams[0])
    print("stream:", streams)

while True:
    samples, timestamp = inlet.pull_sample()
    print("samples:", samples[0])
    print("timestamp:", timestamp)
    time.sleep(0.1)