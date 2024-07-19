import cv2
import os
import sys


def initialize_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        sys.exit(1)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    return cap, out


if __name__ == '__main__':
    video_path = "one1.mp4"  # Update this path
    output_path = 'sonuc.mp4'
    cap, out = initialize_video(video_path, output_path)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    left_margin = int(w * 0.25)
    right_margin = int(w * 0.75)
    top_margin = int(h * 0.1)
    bottom_margin = int(h * 0.9)
    box_thickness = 3
    border_color = (255, 255, 255)
    center_point = (w // 2, h // 2)  # Calculate center point
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video stream closed cleanly")