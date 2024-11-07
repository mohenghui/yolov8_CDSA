# coding:utf-8

import os
import random
import argparse
import xml.etree.ElementTree as ET
from os.path import join

def get_image_extensions(images_folder):
    extensions = set()
    for file in os.listdir(images_folder):
        if os.path.isfile(join(images_folder, file)):
            ext = file.split('.')[-1]
            extensions.add(ext)
    return list(extensions)

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id, annotations_folder, labels_folder, classes):
    image_id=os.path.splitext(image_id.split('/')[-1])[0]
    print(image_id)
    xml_file = join(annotations_folder, '%s.xml' % image_id)
    if os.path.isfile(xml_file):
        in_file = open(xml_file)
        out_file = open(join(labels_folder, '%s.txt' % image_id), 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            # difficult = obj.find('difficult').text
            cls = obj.find('name').text
            # if cls not in classes or int(difficult) == 1:
            #     continue
            if cls not in classes:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
def read_image_paths(dataset_path, image_set):
    file_path = join(dataset_path, f'{image_set}.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        image_ids = [line.strip() for line in file.readlines()]
    return image_ids
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_80', type=str, help='Dataset path')
    opt = parser.parse_args()

    dataset_path = opt.dataset_path
    annotations_folder = join(dataset_path, 'Annotations')
    labels_folder = join(dataset_path, 'labels')
    images_folder = join(dataset_path, 'images')
    sets = ['train', 'test', 'val']
    # classes = ['E2', "J20", "B2", "F14", "Tornado", "F4", "B52", "JAS39", "Mirage2000"]
    classes = ['TYLCV']
    image_extensions = get_image_extensions(images_folder)

    # Split dataset
    trainval_percent = 1.0
    train_percent = 0.9
    xmlfilepath = annotations_folder
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    file_train = open(join(dataset_path, 'train.txt'), 'w')
    file_val = open(join(dataset_path, 'val.txt'), 'w')
    file_test = open(join(dataset_path, 'test.txt'), 'w')
    for i in list_index:
        name = total_xml[i][:-4]
        
        for ext in image_extensions:
            image_file = join(images_folder, name + '.' + ext)
            if os.path.isfile(image_file):
                if i in train:
                    file_train.write(image_file + '\n')
                else:
                    file_val.write(image_file + '\n')
                break

    file_train.close()
    file_val.close()
    file_test.close()
    # Convert annotations
    for image_set in sets:
        if not os.path.exists(labels_folder):
            os.makedirs(labels_folder)
        # image_ids = open(join(dataset_path, '%s.txt' % image_set)).read()
        image_ids=read_image_paths(dataset_path,image_set)
        for image_id in image_ids:
            convert_annotation(image_id, annotations_folder, labels_folder, classes)

if __name__ == '__main__':
    main()
 