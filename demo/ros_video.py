# import cv2
# print(cv2.getBuildInformation())
# this for test = gst-launch-1.0 udpsrc port=5600 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264" ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink
import cv2

# GStreamer pipeline for receiving video over UDP
gst_pipeline = (
    "udpsrc port=5600 ! "
    "application/x-rtp, encoding-name=H264 ! "
    "rtph264depay ! avdec_h264 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Failed to open video stream!")
else:
    print("Video stream opened successfully!")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from the video stream")
            break
        
        cv2.imshow('Video Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
