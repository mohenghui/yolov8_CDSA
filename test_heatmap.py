from pathlib import Path
# import torch
import cv2
from ultralytics import YOLO  # 确保已正确安装Ultralytics YOLO库
from collections import defaultdict

import cv2
import numpy as np

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class Heatmap:
    """A class to draw heatmaps in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the heatmap class with default values for Visual, Image, track, count and heatmap parameters."""

        # Visual information
        self.annotator = None
        self.view_img = False
        self.shape = "circle"

        # Image information
        self.imw = None
        self.imh = None
        self.im0 = None
        self.view_in_counts = True
        self.view_out_counts = True

        # Heatmap colormap and heatmap np array
        self.colormap = None
        self.heatmap = None
        self.heatmap_alpha = 0.5

        # Predict/track information
        self.boxes = None
        self.track_ids = None
        self.clss = None
        self.track_history = defaultdict(list)

        # Region & Line Information
        self.count_reg_pts = None
        self.counting_region = None
        self.line_dist_thresh = 15
        self.region_thickness = 5
        self.region_color = (255, 0, 255)

        # Object Counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Decay factor
        self.decay_factor = 0.99

        # Check if environment support imshow
        # self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        imw,
        imh,
        colormap=cv2.COLORMAP_JET,
        heatmap_alpha=0.5,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        count_reg_pts=None,
        count_txt_thickness=2,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        count_reg_color=(255, 0, 255),
        region_thickness=5,
        line_dist_thresh=15,
        decay_factor=0.99,
        shape="circle",
    ):
        """
        Configures the heatmap colormap, width, height and display parameters.

        Args:
            colormap (cv2.COLORMAP): The colormap to be set.
            imw (int): The width of the frame.
            imh (int): The height of the frame.
            heatmap_alpha (float): alpha value for heatmap display
            view_img (bool): Flag indicating frame display
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            count_reg_pts (list): Object counting region points
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            decay_factor (float): value for removing heatmap area after object passed
            shape (str): Heatmap shape, rect or circle shape supported
        """
        self.imw = imw
        self.imh = imh
        self.heatmap_alpha = heatmap_alpha
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.colormap = colormap

        # Region and line selection
        if count_reg_pts is not None:
            if len(count_reg_pts) == 2:
                print("Line Counter Initiated.")
                self.count_reg_pts = count_reg_pts
                self.counting_region = LineString(count_reg_pts)

            elif len(count_reg_pts) == 4:
                print("Region Counter Initiated.")
                self.count_reg_pts = count_reg_pts
                self.counting_region = Polygon(self.count_reg_pts)

            else:
                print("Region or line points Invalid, 2 or 4 points supported")
                print("Using Line Counter Now")
                self.counting_region = Polygon([(20, 400), (1260, 400)])  # dummy points

        # Heatmap new frame
        self.heatmap = np.zeros((int(self.imh), int(self.imw)), dtype=np.float32)

        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.decay_factor = decay_factor
        self.line_dist_thresh = line_dist_thresh
        self.shape = shape

        # shape of heatmap, if not selected
        if self.shape not in ["circle", "rect"]:
            print("Unknown shape value provided, 'circle' & 'rect' supported")
            print("Using Circular shape now")
            self.shape = "circle"

    def extract_results(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.track_ids = tracks[0].boxes.id.int().cpu().tolist()

    def generate_heatmap(self, im0, tracks):
        """
        Generate heatmap based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            self.heatmap = np.zeros((int(self.imh), int(self.imw)), dtype=np.float32)
            if self.view_img and self.env_check:
                self.display_frames()
            return im0
        self.heatmap *= self.decay_factor  # decay factor
        self.extract_results(tracks)
        self.annotator = Annotator(self.im0, self.count_txt_thickness, None)

        if self.count_reg_pts is not None:
            # Draw counting region
            if self.view_in_counts or self.view_out_counts:
                self.annotator.draw_region(
                    reg_pts=self.count_reg_pts, color=self.region_color, thickness=self.region_thickness
                )

            for box, cls, track_id in zip(self.boxes, self.clss, self.track_ids):
                if self.shape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0 : self.heatmap.shape[0], 0 : self.heatmap.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += (
                        2 * mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    )

                else:
                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += 2

                # Store tracking hist
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                if len(track_line) > 30:
                    track_line.pop(0)

                # Count objects
                if len(self.count_reg_pts) == 4:
                    if self.counting_region.contains(Point(track_line[-1])) and track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        if box[0] < self.counting_region.centroid.x:
                            self.out_counts += 1
                        else:
                            self.in_counts += 1

                elif len(self.count_reg_pts) == 2:
                    distance = Point(track_line[-1]).distance(self.counting_region)
                    if distance < self.line_dist_thresh and track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        if box[0] < self.counting_region.centroid.x:
                            self.out_counts += 1
                        else:
                            self.in_counts += 1
        else:
            for box, cls in zip(self.boxes, self.clss):
                if self.shape == "circle":
                    center = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    radius = min(int(box[2]) - int(box[0]), int(box[3]) - int(box[1])) // 2

                    y, x = np.ogrid[0 : self.heatmap.shape[0], 0 : self.heatmap.shape[1]]
                    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2

                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += (
                        2 * mask[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    )

                else:
                    self.heatmap[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] += 2

        # Normalize, apply colormap to heatmap and combine with original image
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), self.colormap)

        incount_label = f"In Count : {self.in_counts}"
        outcount_label = f"OutCount : {self.out_counts}"

        # Display counts based on user choice
        counts_label = None
        if not self.view_in_counts and not self.view_out_counts:
            counts_label = None
        elif not self.view_in_counts:
            counts_label = outcount_label
        elif not self.view_out_counts:
            counts_label = incount_label
        else:
            counts_label = f"{incount_label} {outcount_label}"

        if self.count_reg_pts is not None and counts_label is not None:
            self.annotator.count_labels(
                counts=counts_label,
                count_txt_size=self.count_txt_thickness,
                txt_color=self.count_txt_color,
                color=self.count_color,
            )

        self.im0 = cv2.addWeighted(self.im0, 1 - self.heatmap_alpha, heatmap_colored, self.heatmap_alpha, 0)

        if self.env_check and self.view_img:
            self.display_frames()

        return self.im0

    def display_frames(self):
        """Display frame."""
        cv2.imshow("Ultralytics Heatmap", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


def process_images(weights_path, source_folder, destination_folder):
    """
    处理图片文件夹中的图片，使用YOLO模型进行目标检测，并生成热力图保存到指定文件夹。
    
    参数:
    - weights_path: YOLO模型权重文件的路径。
    - source_folder: 包含图片的源文件夹路径。
    - destination_folder: 保存热力图的目标文件夹路径。
    """
    # 加载模型
    model = YOLO(weights_path)
    
    # 创建Heatmap实例
    heatmap = Heatmap()
    
    # 确保目标文件夹存在
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)
    
    # 遍历源文件夹中的图片
    for img_path in Path(source_folder).glob('*.png'):  # 假设图片为.jpg格式，根据需要修改
        # 读取图片
        img = cv2.imread(str(img_path))
        
        # 进行目标检测
        results = model(img)
        
        # 设置图片和Heatmap相关参数
        imh, imw = img.shape[:2]
        heatmap.set_args(imw=imw, imh=imh, view_img=False)
        
        # 生成热力图
        heatmap_img = heatmap.generate_heatmap(img, results)
        
        # 保存热力图到目标文件夹
        save_path = destination / img_path.name
        cv2.imwrite(str(save_path), heatmap_img)
        print(f"Saved heatmap for {img_path.name} to {save_path}")

# 设置权重路径、源文件夹路径和目标文件夹路径
weights_path = '/root/autodl-tmp/ultralytics-main/runs/detect/train12/weights/epoch30.pt'
source_folder = '/root/autodl-tmp/ultralytics-main/cam_test_img'
destination_folder = './cam_result1'

# 处理图片并保存热力图
process_images(weights_path, source_folder, destination_folder)