import gi
gi.require_version('Gst', '1.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gst, GLib
import cv2
import numpy as np

# Initialize GStreamer
Gst.init(None)

# Define the pipeline
pipeline = Gst.parse_launch(
    "udpsrc port=5600 ! "
    "application/x-rtp, encoding-name=H264 ! "
    "rtph264depay ! decodebin ! "
    "videoconvert ! appsink name=sink"
)

# Get the appsink
appsink = pipeline.get_by_name('sink')
appsink.set_property('emit-signals', True)
appsink.set_property('sync', False)

def on_new_sample(sink):
    sample = sink.emit('pull-sample')
    buf = sample.get_buffer()
    caps = sample.get_caps()
    arr = np.ndarray(
        (caps.get_structure(0).get_value('height'),
         caps.get_structure(0).get_value('width'),
         3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8
    )
    cv2.imshow('Video Stream', arr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return Gst.FlowReturn.EOS
    return Gst.FlowReturn.OK

appsink.connect('new-sample', on_new_sample)

# Start playing the pipeline
pipeline.set_state(Gst.State.PLAYING)

try:
    loop = GLib.MainLoop()
    loop.run()
except KeyboardInterrupt:
    pass

# Stop the pipeline
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
