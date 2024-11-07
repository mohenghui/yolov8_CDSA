import os
import shutil

def copy_images_by_mod(source_folder, destination_folder, mod_value=20):
    """
    从源文件夹中按取模方式复制图片到目标文件夹。

    参数:
    - source_folder: 源文件夹路径。
    - destination_folder: 目标文件夹路径。
    - mod_value: 取模值，默认为200。
    """
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)
    
    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    # 按取模复制文件
    for i, file in enumerate(files):
        if i % mod_value == 0:  # 每200个文件中复制一个
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy(source_path, destination_path)
            print(f"Copied: {file}")

source_folder = '/root/autodl-tmp/ultralytics-main/datasets/mydata/images'  # 源文件夹路径
destination_folder = '/root/autodl-tmp/ultralytics-main/cam_test_img'  # 目标文件夹路径

copy_images_by_mod(source_folder, destination_folder)
