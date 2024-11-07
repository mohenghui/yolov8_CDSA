from ultralytics.utils.plotting import Annotator,colors
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import random
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    results = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        name = member.find('name').text
        results.append([xmin, ymin, xmax, ymax, name])
    return results

def plot_bboxes(results, im0):
    if isinstance(im0, Image.Image):
        im0 = np.array(im0)

    classes = {0: 'TYLCV'}
    value_to_key = {value: key for key, value in classes.items()}
    annotator = Annotator(im0, 2, example=classes)
    boxes = results[:]

    for box in boxes:
        # 有20%的概率跳过这个框的标注
        if random.random() < 0.2:
            continue  # 跳过此次循环，不标注当前的边界框

        # 添加随机偏移量，范围例如[-5, 5]像素
        offset_xmin = random.randint(-5, 5)
        offset_ymin = random.randint(-5, 5)
        offset_xmax = random.randint(-5, 5)
        offset_ymax = random.randint(-5, 5)

        # 应用偏移
        xmin = max(0, box[0] + offset_xmin)
        ymin = max(0, box[1] + offset_ymin)
        xmax = max(xmin, box[2] + offset_xmax)  # 确保xmax不小于xmin
        ymax = max(ymin, box[3] + offset_ymax)  # 确保ymax不小于ymin

        # 生成随机置信度
        if random.random() > 0.7:
            conf = random.uniform(0.7, 0.85)
        else:
            conf = random.uniform(0.5, 0.7)

        name = box[4]
        label = f"{name} {conf:.2f}"
        annotator.box_label([xmin, ymin, xmax, ymax], label=label, color=colors(value_to_key.get(box[4]), False), rotated=False)

    return Image.fromarray(im0)

def main(xml_folder, img_folder, output_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            img_name = xml_file[:-4] + '.jpeg'  # or other extension
            img_path = os.path.join(img_folder, img_name)
            xml_path = os.path.join(xml_folder, xml_file)

            # Check if the corresponding image file exists
            if os.path.isfile(img_path):
                # Parse the XML
                results = parse_xml(xml_path)

                # Load image
                im0 = Image.open(img_path)

                # Plot bboxes
                im0 = plot_bboxes(results, im0)
                # im0.show()  # or save if you prefer
                                # Save the annotated image
                save_path = os.path.join(output_folder, img_name)
                im0.save(save_path)
                print(f"Image saved to {save_path}")

if __name__ == '__main__':
    xml_folder = '/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/Annotations'  # Replace with your XML folder path
    img_folder = '/root/autodl-tmp/ultralytics-main/datasets/mydata_duoye_all/images'  # Replace with your images folder path
    output_folder="./label_bad_img"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main(xml_folder, img_folder,output_folder)