import os
import shutil
root_path="/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye/images"
annotation_path="/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye/Annotations"
save_path="/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/pass_images"
save_file=[]
for i in os.listdir(annotation_path):
    save_file.append(os.path.basename(i).split('.')[0])
# print(save_file)
for i in os.listdir(root_path):
    old_path=os.path.join(root_path,i)
    new_path=os.path.join(save_path,i)
    filename=i.split('.')[0]
    # print(filename)
    if filename not  in save_file:
        shutil.move(old_path,new_path)
        