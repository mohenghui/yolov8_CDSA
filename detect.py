import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/ultralytics-main/runs/detect/train_jieduan2/weights/epoch125.pt') # select your model.pt path
    model.predict(source='/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/images',
                  imgsz=640,
                  project='runs/detect',
                  name='predictn_best_exp',
                  save=True,
                  # conf=0.2,
                  visualize=True # visualize model features maps
                )