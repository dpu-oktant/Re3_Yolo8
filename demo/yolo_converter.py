from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("best.pt")

# Export the model
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("best.engine")

# Run inference
results = trt_model("plane_test.jpg")