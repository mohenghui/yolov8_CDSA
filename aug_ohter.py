import albumentations as A
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
import shutil
def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    labels = []
    for member in root.findall('object'):
        
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        if xmax <= xmin or ymax <= ymin:
            continue  # Skip invalid boxes
        labels.append(member.find('name').text)
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes, labels



def create_augmentations():
    return [
        A.HorizontalFlip(p=1),
        A.RandomBrightnessContrast(p=1),
        A.GaussNoise(p=1),
        A.VerticalFlip(p=1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        
    ]

def correct_bboxes(bboxes):
    corrected_bboxes = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        # 修正可能的坐标错误
        xmin, xmax = sorted([xmin, xmax])
        ymin, ymax = sorted([ymin, ymax])
        corrected_bboxes.append([xmin, ymin, xmax, ymax])
    return corrected_bboxes

def apply_augmentations(image, bboxes, labels, augmentation_list):
    bboxes = correct_bboxes(bboxes)  # 确保边界框有效
    augmented_results = []
    for augmentation in augmentation_list:
        # 注意：这里必须确保每次调用Compose时都设置bbox_params
        transform = A.Compose([augmentation], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        augmented = transform(image=image, bboxes=bboxes, labels=labels)
        augmented_results.append(augmented)
    return augmented_results



def update_xml(xml_file, transformed_bboxes, labels, output_xml_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall('object')
    if len(objects) != len(transformed_bboxes) or len(objects) != len(labels):
        print("错误：对象数量与标签数量不匹配。")
        return False
    
    for obj, bbox, label in zip(objects, transformed_bboxes, labels):
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(int(bbox[0]))
        bndbox.find('ymin').text = str(int(bbox[1]))
        bndbox.find('xmax').text = str(int(bbox[2]))
        bndbox.find('ymax').text = str(int(bbox[3]))
        obj.find('name').text = label

    tree.write(output_xml_path)
    return True


def process_images(image_folder, annotation_folder, output_image_folder, output_annotation_folder):
    augmentation_list = create_augmentations()
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_annotation_folder):
        os.makedirs(output_annotation_folder)
    
    for filename in os.listdir(image_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff']:
            xml_file = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(annotation_folder, xml_file)
            image_path = os.path.join(image_folder, filename)
            
            # Copy original image and XML to the output folders
            # shutil.copy(image_path, os.path.join(output_image_folder, filename))
            # shutil.copy(xml_path, os.path.join(output_annotation_folder, xml_file))
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, labels = parse_annotation(xml_path)
            
            augmented_images = apply_augmentations(image, bboxes, labels, augmentation_list)
            for idx, augmented in enumerate(augmented_images):
                aug_image = augmented['image']
                transformed_bboxes = augmented['bboxes']
                output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(filename)[0]}_{idx+1}{ext}")
                output_xml_path = os.path.join(output_annotation_folder, f"{os.path.splitext(xml_file)[0]}_{idx+1}.xml")
                
                if update_xml(xml_path, transformed_bboxes, labels, output_xml_path):
                    cv2.imwrite(output_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                # else:
                #     # 删除由于标签不匹配而不保存的图像和标签文件
                #     os.remove(output_image_path)
                #     os.remove(output_xml_path)
if __name__ == "__main__":
    image_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/images'
    annotation_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/Annotations'
    output_image_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye/images'
    output_annotation_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye/Annotations'
    process_images(image_folder, annotation_folder, output_image_folder, output_annotation_folder)