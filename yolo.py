from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to CoreML format
# creates 'yolo11n.mlpackage'
model.export(format="coreml", nms=True)

# Load the exported CoreML model
coreml_model = YOLO("yolo11n.mlpackage", task="detect")

# Run inference
results = coreml_model("https://ultralytics.com/images/bus.jpg")
