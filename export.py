from ultralytics import YOLO

# Load a model
model = YOLO(r'/root/autodl-tmp/ultralytics-main/runs/detect/train8/weights/best.pt')  # load an official model


# Export the model
# model.export(format='onnx')
model.export(format='engine')