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
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get frame from the video stream")
                break
            
            cv2.imshow('Video Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video stream closed cleanly")

