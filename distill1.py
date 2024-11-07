import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        # 'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
        'model': '/root/autodl-tmp/ultralytics-main/runs/prune/yolov8-lamp-exp1-finetune/weights/best.pt',
        'data':r'/root/autodl-tmp/ultralytics-main/meikaung.yaml',
        'imgsz': 640,
        'epochs': 100,
        'batch': 128,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'project':'runs/distill',
        'name':'yolov8n-exp1',
        
        # distill
        'prune_model': True,
        'teacher_weights': '/root/autodl-tmp/ultralytics-main/runs/detect/train10/weights/best.pt',
        'teacher_cfg': './ultralytics/cfg/models/v8/yolov8l.yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '15,18,21',
        'student_kd_layers': '15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    model.distill()