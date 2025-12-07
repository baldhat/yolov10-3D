from pathlib import Path
import numpy as np
import torch
import cv2 as cv
import os as os
import math
import operator
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

from ultralytics.data.datasets.kitti_utils import Object3d, Calibration, affine_transform
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.plotting import KITTIVisualizer, VisObject3D
from scipy.spatial.transform import Rotation

from ultralytics.utils.ops import  xyxy2xywh, xywh2xyxy
from scipy.optimize import linear_sum_assignment

plotter = KITTIVisualizer()

def to_color(a):
    return np.array([int(a[i:i+2], 16) for i in range(0, len(a), 2)]) / 255


gt_color = to_color("52B69A") # Green
our_color = to_color("FFCA3A") # Yellow
base_color = to_color("FF595E") # Red
fov_color = to_color("805D9340") # Purple
text_color = to_color("000000")

colors = plt.get_cmap("tab10")

class Detection3d:
    def __init__(self, line):
        elements = line.split(" ")
        self.classname = elements[0]
        self.alpha = float(elements[3])
        self.bbox = np.array([float(it) for it in elements[4:8]])
        self.dimensions = np.array([float(it) for it in elements[8:11]]) 
        self.location = np.array([float(it) for it in elements[11:14]]) # x,y,z
        self.ry = float(elements[14])
        self.score = float(elements[15])

    
def load_dets(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        return [Detection3d(it) for it in lines]
    
def filter_(dets):
    return [det for det in dets if det.score > 0.3 and det.classname in ["Car", "Pedestrian", "Cyclist", "Van"]]


def center_inside_image(obj, calib):
    # process 2d bbox & get 2d center
    bbox_2d = obj.box2d.copy()

    bbox_2d_ = np.copy(bbox_2d)
    bbox_2d_[:2] = bbox_2d[:2]
    bbox_2d_[2:] = bbox_2d[2:]
    bbox_2d_ = xyxy2xywh(bbox_2d_)

    # process 3d bbox & get 3d center
    center_3d = obj.pos + [0, -obj.h / 2, 0]  # real 3D center in 3D space
    r_center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
    center_3d, _ = calib.rect_to_img(r_center_3d)  # project 3D center to image plane
    center_3d = center_3d[0]  # shape adjustment

    # generate the center of gaussian heatmap [optional: 3d center or 2d center]
    center_heatmap = center_3d.astype(np.int32)
    if center_heatmap[0] < 0 or center_heatmap[0] >= 1280: return False
    if center_heatmap[1] < 0 or center_heatmap[1] >= 384: return False
    return True

def load_calib(path):
    return Calibration(str(path))

def load_image(path):
    return cv.imread(str(path))

def plot_dets(img, dets, calib):
    objects = []
    for i, object in enumerate(dets):
        cls = object.classname
        bbox2d = object.bbox
        dimensions = object.dimensions[::-1]
        translation = object.location
        ry = object.ry
        egoc_rot_matrix = plotter.get_egoc_rot_matrix(ry)

        objects.append( VisObject3D(translation, Rotation.from_matrix(egoc_rot_matrix).as_rotvec(),
                                        dimensions, bbox2d, cls))
    objects = sorted(objects, key=lambda x: x.translation[2], reverse=True)
    plotter.plot_3d_obj(img, objects, calib.P2, [colors(1) for _ in objects])

def plot_bev(our_dets, filename, fov=60):
    plt.clf()

    def get_rotated_rectangle_points(center, size, angle_degrees):
        cx, cy = center
        w, h = size
        angle = np.deg2rad(angle_degrees)

        # Rectangle corners before rotation (centered at origin)
        rect = np.array([
            [-w/2, -h/2],
            [ w/2, -h/2],
            [ w/2,  h/2],
            [-w/2,  h/2]
        ])

        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        # Rotate and translate
        rotated_rect = rect @ R.T + [cx, cy]
        return rotated_rect

    fig, ax = plt.subplots(1, 1,
                        figsize=(24, 12), gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)

    num_lines = 11
    R = 50
    border = 3
    ax.set_xlim(-R - border, R + border)
    ax.set_ylim(-border, R + border)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((1, 1, 1))
    

    # for theta in np.linspace(0, np.pi, 7):
    #     xs, ys = [R * np.cos(theta), 0], [R * np.sin(theta), 0]
    #     ax.plot(xs, ys, linewidth=2, color=(1, 1, 1), zorder=1)

    for radius, c_color in zip(np.linspace(R, 0, num_lines), np.linspace(0.95, 0.5, num_lines)):
        x = np.sin(np.deg2rad(fov / 2)) * (radius - 1.5)
        y = np.cos(np.deg2rad(fov / 2)) * (radius - 1.5)
        if radius % 10 == 0:
            ax.text(x + 1.3, y - 1.2, str(int(radius)) + "m", rotation=-(5 + fov/2), fontsize=25, color=(0.15, 0.15, 1))
        if radius == 0:
            continue
        #circle = Circle((0, 0), radius, color=(0, 0, 0), linewidth=3, fill=False, zorder=1)
        circle = Circle((0, 0), radius, color=(c_color, c_color, c_color), linewidth=3, fill=True, zorder=1)
        ax.add_artist(circle)
        
        
    lightblue = (0, 252/255.0, 239/255.0)
    wedge = Wedge((0, 0), R, -fov/2 + 90, fov/2 + 90, 
                  color=fov_color,  
                  #linewidth=3, 
                  fill=True)
    ax.add_artist(wedge)
        
    for j, object in enumerate(our_dets):
        dimensions = object.dimensions[::-1][:2]
        translation = object.location[[0, 2]]
        ry = -object.ry

        corners = get_rotated_rectangle_points(translation, dimensions, ry * 180 / np.pi)
        art = ax.add_artist(Polygon(corners, closed=True, fill=False, edgecolor=our_color, facecolor=our_color, zorder=3, linewidth=5))
        if j == 0:
            art.set_label("Ours")

    plt.savefig(filename, bbox_inches="tight", format="svg")
    fig.clear()
    plt.close()

def plot_all(img, our_dets, calib, out_path):
    our_img = img.copy()
        
    # plot_labels(our_img, gts, calib, color="g")
    plot_dets(our_img, our_dets, calib)
    our_name = out_path.replace(".png", "_ours.png")
    cv.imwrite(our_name, (our_img*255.0).astype(np.uint8))
    
    # plot_labels(base_img, gts, calib, color="g")
    # plot_dets(base_img, base_dets, calib, color="r")
    # base_name = out_path.replace(".png", "_base.png")
    # cv.imwrite(base_name, (base_img*255.0).astype(np.uint8))
    # print(base_name)
    
    plot_bev(our_dets, out_path.replace(".png", "_bev.svg"), np.rad2deg(2*np.arctan2(our_img.shape[1], 2* calib.fu)))
    
    
def create(ours_path, gt_path):
    ours_name = str(ours_path).split("/")[-1]
    name = str(np.random.randint(0, 10000))
    output_path = Path(f"/usr/wiss/mejo/storage/user/_archiv_paper/2026_CVPR_LeAD-M3D/von_johannes/yolov10-3D_x2_vis/waymo_video_dir_{name}") / ours_name
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    gt_path = Path(gt_path)

    from tqdm import tqdm
    for fn in tqdm(os.listdir(gt_path)):
        filename = fn.strip().split(".")[0][4:] + ".txt"
        # load dets and gts
        our_dets = load_dets(ours_path / "preds" / filename)
       
        calib = load_calib(gt_path / "../../../calib_cam_to_cam.txt")

        # filter dets by score and class
        our_dets_ = filter_(our_dets)
        # if len(our_dets_) == 0:
        #     continue
        
        img = load_image(gt_path / fn).astype(np.float32) / 255.0
        out_path = output_path / fn
        plot_all(img, our_dets_, calib, str(out_path))
    
    # 1. Create video for original *.png files
    png_res="1280x384"; p = Path(output_path)
    svg_res="1024x546"
    os.system(f"cd {output_path} && ffmpeg -framerate 10 -pattern_type glob -i '*.png' -s {png_res} -c:v libx264 -pix_fmt yuv420p out_png.mp4")
    
    # 2. Delete all *.png files (original and converted SVGs share the same logic now for ultimate minimalism)
    [os.remove(f) for f in p.iterdir() if f.is_file() and f.suffix == '.png']
    
    # 3. Convert all *.svg to *.png (rasterization at final desired resolution)
    [os.system(f"cd {output_path} && ffmpeg -i '{svg.name}' -s {svg_res} '{svg.stem}.png'") for svg in p.glob('*.svg')]
    
    # 4. Create video from the newly created *.png files (which were originally SVGs)
    os.system(f"cd {output_path} && ffmpeg -framerate 10 -pattern_type glob -i '*.png' -s {svg_res} -c:v libx264 -pix_fmt yuv420p out_svg.mp4")
    
    # BONUS: Clean up the intermediate *.svg files and the newly created *.png files
    [os.remove(f) for f in p.iterdir() if f.is_file() and f.suffix in ('.svg', '.png')]
    print(output_path / "out.mp4")
        
    
if __name__=='__main__':
    test_plot = False

    if len(sys.argv) >= 1:
        ours_path = Path(sys.argv[1])
    else:
        raise RuntimeError("no arguments")

    create(ours_path, "/storage/group/deepscenario/KITTI/kitti_raw_data/2011_09_26/2011_09_26_drive_0039_sync/image_02/data/")