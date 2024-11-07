from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO("./yolov8n.pt")  # 从头开始构建新模型

    # Use the model
    results = model.train(data="./TYLCV_duoye_80.yaml", epochs=300, device='0', workers=16,batch=64, seed=42,patience=100,task='detect', project='runs/detect',mode='train',optimizer="SGD",save_period=25)  # 训练模
