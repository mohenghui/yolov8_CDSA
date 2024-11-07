import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': 'runs/train/yolov8-DyHead/weights/best.pt',
        'data':'/root/data_ssd/dataset_crowdhuman/data_20per.yaml',
        'imgsz': 640,
        'epochs': 250,
        'batch': 16,
        'workers': 8,
        'cache': True,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'yolov8-DyHead-lamp-exp3',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 3.0,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': None
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)