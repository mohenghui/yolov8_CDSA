from ultralytics import YOLO
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO("./yolov8n.pt")  # 从头开始构建新模型

    # Use the model
    results = model.train(data="./meikaung.yaml", epochs=300, device='0', batch=-1, seed=42,patience=100,task='detect', mode='train',save_period=30)  # 训练模
