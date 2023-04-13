from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=5, batch=10, imgsz = 640)  # train the model
