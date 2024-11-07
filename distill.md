# Distill Experiment (For BiliBili魔鬼面具)

### student:yolov8n teacher:yolov8s Dataset:Visdrone 30% Training Data

```
CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log

CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8s.log 2>&1 & tail -f logs/yolov8s.log
nohup python val.py > logs/yolov8s-test.log 2>&1 & tail -f logs/yolov8s-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-visdrone-cwd-exp1.log 2>&1 & tail -f logs/yolov8-visdrone-cwd-exp1.log
nohup python val.py > logs/yolov8n-cwd-exp1-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 1.5
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-cwd-exp2.log 2>&1 & tail -f logs/yolov8-cwd-exp2.log
nohup python val.py > logs/yolov8n-cwd-exp2-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '12,15,18,21',
    'student_kd_layers': '12,15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-cwd-exp3.log 2>&1 & tail -f logs/yolov8-cwd-exp3.log 
nohup python val.py > logs/yolov8n-cwd-exp3-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp3-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'AdamW',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-mgd-exp1',
    'lr0': 0.001,
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.2
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-mgd-exp1.log 2>&1 & tail -f logs/yolov8-mgd-exp1.log
nohup python val.py > logs/yolov8n-mgd-ex1-test.log 2>&1 & tail -f logs/yolov8n-mgd-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'AdamW',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-mgd-exp2',
    'lr0': 0.001,
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.4
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-mgd-exp2.log 2>&1 & tail -f logs/yolov8-mgd-exp2.log
nohup python val.py > logs/yolov8n-mgd-exp2-test.log 2>&1 & tail -f logs/yolov8n-mgd-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-mimic-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-mimic-exp1.log 2>&1 & tail -f logs/yolov8-mimic-exp1.log 
nohup python val.py > logs/yolov8n-mimic-exp1-test.log 2>&1 & tail -f logs/yolov8n-mimic-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-mimic-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 2.0
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-mimic-exp2.log 2>&1 & tail -f logs/yolov8-mimic-exp2.log 
nohup python val.py > logs/yolov8n-mimic-exp2-test.log 2>&1 & tail -f logs/yolov8n-mimic-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-l2-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 0.5,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-l2-exp1.log 2>&1 & tail -f logs/yolov8-l2-exp1.log 
nohup python val.py > logs/yolov8n-l2-exp1-test.log 2>&1 & tail -f logs/yolov8n-l2-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.5,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-BCKD-exp1.log 2>&1 & tail -f logs/yolov8-BCKD-exp1.log 
nohup python val.py > logs/yolov8n-BCKD-exp1-test.log 2>&1 & tail -f logs/yolov8n-BCKD-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.8,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-BCKD-exp2.log 2>&1 & tail -f logs/yolov8-BCKD-exp2.log 
nohup python val.py > logs/yolov8n-BCKD-exp2-test.log 2>&1 & tail -f logs/yolov8n-BCKD-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.4,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-BCKD-exp3.log 2>&1 & tail -f logs/yolov8-BCKD-exp3.log 
nohup python val.py > logs/yolov8n-BCKD-exp3-test.log 2>&1 & tail -f logs/yolov8n-BCKD-exp3-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-exp4',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.3,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'mimic',
    'feature_loss_ratio': 1.0
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-BCKD-exp4.log 2>&1 & tail -f logs/yolov8-BCKD-exp4.log 
nohup python val.py > logs/yolov8n-BCKD-exp4-test.log 2>&1 & tail -f logs/yolov8n-BCKD-exp4-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-CWD-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.3,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.7
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-BCKD-CWD-exp1.log 2>&1 & tail -f logs/yolov8-BCKD-CWD-exp1.log 
nohup python val.py > logs/yolov8n-BCKD-CWD-exp1-test.log 2>&1 & tail -f logs/yolov8n-BCKD-CWD-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-CWD-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.6,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.7
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-BCKD-CWD-exp2.log 2>&1 & tail -f logs/yolov8-BCKD-CWD-exp2.log 
nohup python val.py > logs/yolov8n-BCKD-CWD-exp2-test.log 2>&1 & tail -f logs/yolov8n-BCKD-CWD-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-CWD-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.8,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.85
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-BCKD-CWD-exp3.log 2>&1 & tail -f logs/yolov8-BCKD-CWD-exp3.log 
nohup python val.py > logs/yolov8n-BCKD-CWD-exp3-test.log 2>&1 & tail -f logs/yolov8n-BCKD-CWD-exp3-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32, 
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-CWD-exp4',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.85
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8-BCKD-CWD-exp4.log 2>&1 & tail -f logs/yolov8-BCKD-CWD-exp4.log 
nohup python val.py > logs/yolov8n-BCKD-CWD-exp4-test.log 2>&1 & tail -f logs/yolov8n-BCKD-CWD-exp4-test.log
```

### student:yolov8n-lamp-exp3 teacher:yolov8s Dataset:Visdrone 30% Training Data
```
param_dict = {
    # origin
    'model': 'runs/prune/yolov8n-visdrone-lamp-exp3-prune/weights/prune_notv2.pt',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 20,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp1',
    
    # distill
    'prune_model': True,
    'teacher_weights': 'runs/train/yolov8s-visdrone/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '15,18,21',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.5
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8-visdrone-cwd-exp1.log 2>&1 & tail -f logs/yolov8-visdrone-cwd-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-cwd-exp1-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp1-test.log
```

### student:yolov8n-asf-p2 teacher:yolov8s-asf-p2 Dataset:Visdrone 30% Training Data
```
CUDA_VISIBLE_DEVICES=0 nohup python train.py > logs/yolov8n.log 2>&1 & tail -f logs/yolov8n.log
nohup python val.py > logs/yolov8n-test.log 2>&1 & tail -f logs/yolov8n-test.log

CUDA_VISIBLE_DEVICES=1 nohup python train.py > logs/yolov8s.log 2>&1 & tail -f logs/yolov8s.log
nohup python val.py > logs/yolov8s-test.log 2>&1 & tail -f logs/yolov8s-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.4
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8n-cwd-exp1.log 2>&1 & tail -f logs/yolov8n-cwd-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-cwd-exp1-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,25,28,30',
    'student_kd_layers': '13,17,20,23,25,28,30',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.4
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8n-cwd-exp2.log 2>&1 & tail -f logs/yolov8n-cwd-exp2.log
nohup python val.py > logs/yolov8n-cwd-exp2-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp3',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.6
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8n-cwd-exp3.log 2>&1 & tail -f logs/yolov8n-cwd-exp3.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-cwd-exp3-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp3-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp4',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.2
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8n-cwd-exp4.log 2>&1 & tail -f logs/yolov8n-cwd-exp4.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-cwd-exp4-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp4-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-cwd-exp5',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8n-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'linear_epoch',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.4
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8n-cwd-exp5.log 2>&1 & tail -f logs/yolov8n-cwd-exp5.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-cwd-exp5-test.log 2>&1 & tail -f logs/yolov8n-cwd-exp5-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-mgd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.05
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8n-mgd-exp1.log 2>&1 & tail -f logs/yolov8n-mgd-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-mgd-exp1-test.log 2>&1 & tail -f logs/yolov8n-mgd-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-mgd-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'l2',
    'logical_loss_ratio': 1.0,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.02
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8n-mgd-exp2.log 2>&1 & tail -f logs/yolov8n-mgd-exp2.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-mgd-exp2-test.log 2>&1 & tail -f logs/yolov8n-mgd-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.4,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.05
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8n-BCKD-exp1.log 2>&1 & tail -f logs/yolov8n-BCKD-exp1.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-BCKD-exp1-test.log 2>&1 & tail -f logs/yolov8n-BCKD-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8n-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'kd_loss_type': 'logical',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.4,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'mgd',
    'feature_loss_ratio': 0.05
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8n-BCKD-exp2.log 2>&1 & tail -f logs/yolov8n-BCKD-exp2.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-BCKD-exp2-test.log 2>&1 & tail -f logs/yolov8n-BCKD-exp2-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-cwd-exp1',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.2,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.25
}
CUDA_VISIBLE_DEVICES=0 nohup python distill.py > logs/yolov8n-BCKD-cwd-exp1.log 2>&1 & tail -f logs/yolov8n-BCKD-cwd-exp1.log
CUDA_VISIBLE_DEVICES=0 nohup python val.py > logs/yolov8n-BCKD-cwd-exp1-test.log 2>&1 & tail -f logs/yolov8n-BCKD-cwd-exp1-test.log

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-ASF-P2.yaml',
    'data':'/root/data_ssd/dataset_visdrone/data_exp.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'yolov8n-BCKD-cwd-exp2',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'runs/train/yolov8s-ASF-P2/weights/best.pt',
    'teacher_cfg': 'ultralytics/cfg/models/v8/yolov8s-ASF-P2.yaml',
    'kd_loss_type': 'all',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.4,
    
    'teacher_kd_layers': '13,17,20,23,28',
    'student_kd_layers': '13,17,20,23,28',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.2
}
CUDA_VISIBLE_DEVICES=1 nohup python distill.py > logs/yolov8n-BCKD-cwd-exp2.log 2>&1 & tail -f logs/yolov8n-BCKD-cwd-exp2.log
CUDA_VISIBLE_DEVICES=1 nohup python val.py > logs/yolov8n-BCKD-cwd-exp2-test.log 2>&1 & tail -f logs/yolov8n-BCKD-cwd-exp2-test.log
```

### student:yolov8n-fasternet teacher:yolov8n-fasternet/yolov8n(这个纯粹用于教学换主干的剪枝)
```
param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-fasternet.yaml',
    'data':'dataset/data.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'test',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'teacher_cfg': 'runs/train/yolov8n/weights/best.pt',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.4,
    
    'teacher_kd_layers': '2,4,6,9',
    'student_kd_layers': '0-1,0-2,0-3,0-4',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.2
}

param_dict = {
    # origin
    'model': 'ultralytics/cfg/models/v8/yolov8n-fasternet.yaml',
    'data':'dataset/data.yaml',
    'imgsz': 640,
    'epochs': 250,
    'batch': 8,
    'workers': 8,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 10,
    'project':'runs/distill',
    'name':'test',
    
    # distill
    'prune_model': False,
    'teacher_weights': 'ultralytics/cfg/models/v8/yolov8n.yaml',
    'teacher_cfg': 'runs/train/yolov8n/weights/best.pt',
    'kd_loss_type': 'feature',
    'kd_loss_decay': 'constant',
    
    'logical_loss_type': 'BCKD',
    'logical_loss_ratio': 0.4,
    
    'teacher_kd_layers': '15,18,21',
    'student_kd_layers': '7,10,13',
    'feature_loss_type': 'cwd',
    'feature_loss_ratio': 0.2
}
```

# 使用教程

    蒸馏操作问题、报错问题统一群里问,我群里回复,谢谢~

# 环境

    没什么特别要求,跟正常的v8一样.

# 视频
1. 整体的使用教程(必看!)
链接：https://pan.baidu.com/s/1zy_yfH_ErSdDe88yvTdUcw?pwd=culk 
提取码：culk # BiliBili 魔鬼面具

2. 怎么把蒸馏代码加到v8的代码上
链接：https://pan.baidu.com/s/1MIN2JhqgFx5h1Rga84f86w?pwd=zyhr 
提取码：zyhr # BiliBili 魔鬼面具

3. 更换主干后蒸馏的使用注意点
链接：https://pan.baidu.com/s/1_SjffUbi8ey5f4YXp3dHgw?pwd=la2v 
提取码：la2v # BiliBili 魔鬼面具

4. BCKD蒸馏方法说明
链接：https://pan.baidu.com/s/12GDfcSGuKoyqRvZoxuwQRQ?pwd=dj3t 
提取码：dj3t # BiliBili 魔鬼面具

5. 知识蒸馏中教师学生的一些建议
链接：https://pan.baidu.com/s/10AYI2jL6x2Rm_JbGfUVKrw?pwd=z44l 
提取码：z44l # BiliBili 魔鬼面具

# 我自己跑的实验数据
1. yolov8n-visdrone
链接：https://pan.baidu.com/s/1zweHjEPILJAwJ-QMnUX0ug?pwd=hnqa 
提取码：hnqa # BiliBili 魔鬼面具
2. yolov8n-lamp-visdrone
链接：https://pan.baidu.com/s/1NjLniVXpSeivGIFoR_2ATA?pwd=jlpz 
提取码：jlpz # BiliBili 魔鬼面具
3. yolov8n-asf-p2
链接：https://pan.baidu.com/s/1Y2Auyc8pLJuInpjselj7tw?pwd=da4p 
提取码：da4p # BiliBili 魔鬼面具