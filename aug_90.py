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
def rotate_image_and_bbox(image, bboxes, angle):
    (h, w) = image.shape[:2]
    if angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # new_bboxes = [[bbox[1], w - bbox[2], bbox[3], w - bbox[0]] for bbox in bboxes]
        # new_bboxes = [[bbox[1], w - bbox[2], bbox[3], w - bbox[0]] for bbox in bboxes]
        new_bboxes = [[h - bbox[3], bbox[0], h - bbox[1], bbox[2]] for bbox in bboxes]
    elif angle == -90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # new_bboxes = [[h - bbox[3], bbox[0], h - bbox[1], bbox[2]] for bbox in bboxes]
        new_bboxes = [[bbox[1], w - bbox[2], bbox[3], w - bbox[0]] for bbox in bboxes]
    return rotated_image, new_bboxes


def apply_custom_rotations(image, bboxes, labels):
    rotated_image_90, new_bboxes_90 = rotate_image_and_bbox(image, bboxes, 90)
    rotated_image_neg_90, new_bboxes_neg_90 = rotate_image_and_bbox(image, bboxes, -90)
    return [
        (rotated_image_90, new_bboxes_90, labels),
        (rotated_image_neg_90, new_bboxes_neg_90, labels)
    ]


def update_xml(xml_file, transformed_bboxes, labels, output_xml_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    size.find('width').text = str(height)
    size.find('height').text = str(width)
    objects = root.findall('object')
    for obj, bbox, label in zip(objects, transformed_bboxes, labels):
        bndbox = obj.find('bndbox')
        bndbox.find('xmin').text = str(int(bbox[0]))
        bndbox.find('ymin').text = str(int(bbox[1]))
        bndbox.find('xmax').text = str(int(bbox[2]))
        bndbox.find('ymax').text = str(int(bbox[3]))
        obj.find('name').text = label
    tree.write(output_xml_path)

def process_images(image_folder, annotation_folder, output_image_folder, output_annotation_folder):
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
            
            shutil.copy(image_path, os.path.join(output_image_folder, filename))
            shutil.copy(xml_path, os.path.join(output_annotation_folder, xml_file))
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, labels = parse_annotation(xml_path)
            
            rotated_images = apply_custom_rotations(image, bboxes, labels)
            for idx, (aug_image, transformed_bboxes, transformed_labels) in enumerate(rotated_images):
                output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(filename)[0]}_rot{['90', '-90'][idx]}{ext}")
                output_xml_path = os.path.join(output_annotation_folder, f"{os.path.splitext(xml_file)[0]}_rot{['90', '-90'][idx]}.xml")
                
                update_xml(xml_path, transformed_bboxes, transformed_labels, output_xml_path)
                cv2.imwrite(output_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    image_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/images'
    annotation_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/Annotations'
    output_image_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye/images'
    output_annotation_folder = r'/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye/Annotations'
    process_images(image_folder, annotation_folder, output_image_folder, output_annotation_folder)
