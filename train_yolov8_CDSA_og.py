from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 加载模型
    model = YOLO("/root/autodl-tmp/ultralytics-main/ultralytics/cfg/models/v8/yolov8n.yaml")  # 从头开始构建新模型
    # model.load("./yolov8n.pt")
    # Use the model
    results = model.train(data="./TYLCV_duoye.yaml", epochs=300, device='0', workers=16,batch=64, seed=42,patience=100,task='detect', project='runs/detect',mode='train',optimizer="SGD",save_period=25)  # 训练模
