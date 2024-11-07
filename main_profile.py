import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/root/autodl-tmp/ultralytics-main/runs/prune/yolov8-lamp-exp1-prune3/weights/prune.pt')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()