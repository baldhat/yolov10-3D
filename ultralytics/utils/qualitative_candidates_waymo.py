from pathlib import Path
import numpy as np
import torch
import cv2 as cv
import os
import math
import operator
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

from ultralytics.data.datasets.kitti_utils import Object3d, Calibration
from ultralytics.data.datasets.waymo import WaymoDataset
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.plotting import KITTIVisualizer, VisObject3D
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

plotter = KITTIVisualizer()

def to_color(a):
    return np.array([int(a[i:i+2], 16) for i in range(0, len(a), 2)]) / 255


gt_color = to_color("52B69A") # Green
our_color = to_color("FFCA3A") # Yellow
base_color = to_color("FF595E") # Red
fov_color = to_color("805D9340") # Purple
text_color = to_color("000000")

class Args:
    overfit = False
    fliplr = False
    random_crop = False
    scale = 1.0
    min_scale = False
    max_scale = False
    translate = False
    mixup = False
    max_depth_threshold = False
    min_depth_threshold = False
    load_depth_maps = False
    rotation = False
    virtual_focal_length = False
    
args = Args()
dataset = WaymoDataset("/storage/group/deepscenario/waymo/val.json", "val", args)

class Detection3d:
    def __init__(self, bbox, type, frame_id, score, calib, bs_counter):
        self.classname = dataset.eval_id2cls[type]
        self.location = np.array([float(it) for it in bbox[:3]])
        self.dimensions = np.array([float(it) for it in bbox[3:6]])  # l,w,h
        self.ry = float(bbox[6])
        self.score = float(score)
        center_3d = self.location - [0, self.dimensions[2] / 2, 0] # we need height
        self.bbox = dataset.recompute_bbox_2d(center_3d.reshape(-1, 3), np.copy(self.dimensions[::-1]), self.ry, calib)
        self.line_index = bs_counter

def load_labels(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        return [Object3d(it, idx=i) for i, it in enumerate(lines)]
    
def load_dets(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        return [Detection3d(it) for it in lines]
    
def filter_(dets: [Detection3d]):
    return [det for det in dets if det.score > 0.1 and det.classname in ["Car", "Pedestrian", "Cyclist"]]

def filter_gts(dets):
    return [det for det in dets if det.classname in ["Car", "Pedestrian", "Cyclist"]]

def associate(gts: [Object3d], dets: [Detection3d]):
    if len(dets) == 0:
        return [], [], []
    iou = box_iou(torch.tensor(np.array([it.bbox for it in gts])), torch.tensor(np.array([it.bbox for it in dets])))
    false_positives = list(np.where(np.all(iou.cpu().numpy() == 0, axis=0))[0])
    row_ind, col_ind = linear_sum_assignment(iou.cpu().detach().numpy(), maximize=True)
    matched_gts = [gts[ind_r] for ind_r, ind_c in zip(row_ind, col_ind)]
    matched_dets = [dets[ind_c] for ind_r, ind_c in zip(row_ind, col_ind)]
    return matched_gts, matched_dets, [dets[fp] for fp in false_positives]

def calculate_errors(gts: [Detection3d], dets: [Detection3d]):
    pos_errors = [np.linalg.norm(gt.location - det.location) for gt,det in zip(gts, dets)]
    rot_errors = [np.abs((gt.ry - det.ry)%np.pi) for gt,det in zip(gts, dets)]
    return pos_errors, rot_errors

def equals(gt1: Object3d, gt2: Object3d):
    return gt1.line_index == gt2.line_index

def load_calib(idx):
    return dataset.get_calib(idx)

def load_image(idx):
    return dataset.get_image(idx)

def plot_labels(img, gts: [Detection3d], calib, color):
    for object in gts:
        cls = object.classname
        bbox2d = object.bbox
        dimensions = object.dimensions
        translation = object.location
        ry = object.ry
        egoc_rot_matrix = plotter.get_egoc_rot_matrix(ry)

        plotter.plot_3d_obj(img,
                            VisObject3D(translation, Rotation.from_matrix(egoc_rot_matrix).as_rotvec(),
                                        dimensions, bbox2d, cls),
                            calib.P2, bbox2d=False, gt=False)

def plot_dets(img, dets, calib, color):
    for object in dets:
        cls = object.classname
        bbox2d = object.bbox
        dimensions = object.dimensions
        translation = object.location
        ry = object.ry
        egoc_rot_matrix = plotter.get_egoc_rot_matrix(ry)

        plotter.plot_3d_obj(img,
                            VisObject3D(translation, Rotation.from_matrix(egoc_rot_matrix).as_rotvec(),
                                        dimensions, bbox2d, cls),
                            calib.P2, bbox2d=False, gt=True)

def plot_bev(gts, base_dets, our_dets, filename, fov=60):
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
    R = 100
    border = 3
    ax.set_xlim(-R - border, R + border)
    ax.set_ylim(-border, R + border)
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor((0.9, 0.9, 0.9))
    

    # for theta in np.linspace(0, np.pi, 7):
    #     xs, ys = [R * np.cos(theta), 0], [R * np.sin(theta), 0]
    #     ax.plot(xs, ys, linewidth=2, color=(1, 1, 1), zorder=1)

    for radius, c_color in zip(np.linspace(R, 0, num_lines), np.linspace(0.8, 0.35, num_lines)):
        x = np.sin(np.deg2rad(fov / 2)) * (radius - 1.5)
        y = np.cos(np.deg2rad(fov / 2)) * (radius - 1.5)
        if radius % 10 == 0:
            ax.text(x + 1.3, y - 1.2, str(int(radius)) + "m", rotation=-(5 + fov/2), fontsize=25, color=text_color)
        if radius == 0:
            continue
        circle = Circle((0, 0), radius, color=(c_color, c_color, c_color), linewidth=2, fill=True, zorder=1)
        ax.add_artist(circle)
        
        
    wedge = Wedge((0, 0), R, -fov/2 + 90, fov/2 + 90, color=fov_color)
    ax.add_artist(wedge)

    for j, object in enumerate(gts):            
        dimensions = object.dimensions[:2]
        translation = object.location[[0, 2]]
        ry = object.ry

        corners = get_rotated_rectangle_points(translation, dimensions, ry * 180 / np.pi)
        art = ax.add_artist(Polygon(corners, closed=True, fill=False, edgecolor=gt_color, facecolor=gt_color, zorder=3, linewidth=5))
        if j == 0:
            art.set_label("Ground Truth")
        
    for j, object in enumerate(base_dets):
        dimensions = object.dimensions[:2]
        translation = object.location[[0, 2]]
        ry = object.ry

        corners = get_rotated_rectangle_points(translation, dimensions, ry * 180 / np.pi)
        art = ax.add_artist(Polygon(corners, closed=True, fill=False, edgecolor=base_color, facecolor=base_color, zorder=3, linewidth=5))
        if j == 0:
            art.set_label("Baseline")
        
    for j, object in enumerate(our_dets):
        dimensions = object.dimensions[:2]
        translation = object.location[[0, 2]]
        ry = object.ry

        corners = get_rotated_rectangle_points(translation, dimensions, ry * 180 / np.pi)
        art = ax.add_artist(Polygon(corners, closed=True, fill=False, edgecolor=our_color, facecolor=our_color, zorder=3, linewidth=5))
        if j == 0:
            art.set_label("Ours")

    plt.savefig(filename, bbox_inches="tight", format="svg")
    fig.clear()
    plt.close()
    print(filename)

def plot_all(img, gts, our_dets, base_dets, calib, out_path):
    base_img = img.copy()
    our_img = img.copy()
        
    plot_labels(our_img, gts, calib, color="g")
    plot_dets(our_img, our_dets, calib, color="r")
    our_name = out_path.replace(".png", "_ours.png")
    cv.imwrite(our_name, (our_img*255.0).astype(np.uint8))
    print(our_name)
    
    plot_labels(base_img, gts, calib, color="g")
    plot_dets(base_img, base_dets, calib, color="r")
    base_name = out_path.replace(".png", "_base.png")
    cv.imwrite(base_name, (base_img*255.0).astype(np.uint8))
    print(base_name)
    
    plot_bev(gts, base_dets, our_dets, out_path.replace(".png", "_bev.svg"), np.rad2deg(2*np.arctan2(base_img.shape[1], 2* calib.fu)))
    

base_path = Path("/storage/group/deepscenario/for_jonathan/waymo_baseline_x")
ours_name = "waymo_ours_x"

output_path = Path("/storage/user/mijo/mijo/qualitative") / ours_name
if not os.path.exists(output_path):
    os.mkdir(output_path)

ours_path = Path("/home/stud/mijo/dev/yolov10-3D/runs/detect/" + ours_name)
gt_path = Path("/storage/group/deepscenario/KITTI/training/label_2/")

counter = 0

scores = {}

base = json.load(open(base_path / "eval_results.json", "r"))
b_dets = base["pred"]
gts = base["gt"]
o_dets = json.load(open(ours_path / "eval_results.json", "r"))["pred"]

bbox_bs ,type_bs, frame_id_bs, score_bs = b_dets["bbox"], b_dets["type"], b_dets["frame_id"], b_dets["score"]
bbox_os ,type_os, frame_id_os, score_os = o_dets["bbox"], o_dets["type"], o_dets["frame_id"], o_dets["score"]
bbox_gts ,type_gts, frame_id_gts, score_gts = gts["bbox"], gts["type"], gts["frame_id"], gts["score"]


gt_index = 0
our_index = 0
base_index = 0
for frame_id in range(np.max(np.array(frame_id_gts))):
    calib = load_calib(frame_id)
    
    current_gts = []
    gt_counter = 0
    while frame_id_gts[gt_index] == frame_id:
        it = Detection3d(bbox_gts[gt_index], type_gts[gt_index], frame_id_gts[gt_index], score_gts[gt_index], calib, gt_counter)
        current_gts.append(it)
        gt_index += 1
        gt_counter += 1
    
    current_ours = []
    our_counter = 0
    while frame_id_os[our_index] == frame_id:
        it = Detection3d(bbox_os[our_index], type_os[our_index], frame_id_os[our_index], score_os[our_index], calib, our_counter)
        current_ours.append(it)
        our_index += 1
        our_counter += 1
        
    current_bs = []
    bs_counter = 0
    while frame_id_bs[base_index] == frame_id:
        it = Detection3d(bbox_bs[base_index], type_bs[base_index], frame_id_bs[base_index], score_bs[base_index], calib, bs_counter)
        current_bs.append(it)
        base_index += 1
        bs_counter += 1
    
    plot = False

    # filter dets by score and class
    base_dets_ = filter_(current_bs)
    our_dets_ = filter_(current_ours)
    if len(our_dets_) == 0 or len(current_gts) == 0:
        continue
    
    # associate dets to gts
    base_gts, base_dets, base_false_positives = associate(current_gts, base_dets_)
    our_gts, our_dets, our_false_positives = associate(current_gts, our_dets_)
    
    improvement_counter = 0
    # check missing detections
    if len(base_false_positives) > len(our_false_positives):
        pass
        #print(len(base_false_positives), len(our_false_positives))
        #improvement_counter += (len(base_false_positives) - len(our_false_positives))

    if len(base_dets) > len(our_dets):
        continue
    
    # calculate position and rotation errors
    base_pos_errors, base_rot_errors = calculate_errors(base_gts, base_dets)
    our_pos_errors, our_rot_errors = calculate_errors(our_gts, our_dets)
    
    # print where the base errors are significantly larger than our errors
    for j, our_gt in enumerate(our_gts):
        found = False
        for k, base_gt in enumerate(base_gts):
            found = True
            if not equals(base_gt, our_gt):
                continue
            
            diff = base_pos_errors[k] - our_pos_errors[j]
            if diff > 1 and diff < 12:
                # print(f"Better Location! Base: {base_dets[k].location}, Ours: {our_dets[j].location}")
                improvement_counter += 1 # math.ceil(base_pos_errors[k] - our_pos_errors[j] - 5)
                
            if np.abs(base_rot_errors[k] - our_rot_errors[j]) > 1:
                #print(f"Better Rotation! Base: {base_dets[k].ry}, Ours: {our_dets[j].ry}")
                #plot = True
                pass
        if not found:
            # print("We detected more objects")
            improvement_counter += 1
                
    if improvement_counter > 1:
        img_name = f"{frame_id:06d}.png"
        img = np.array(load_image(frame_id)).astype(np.float32)[:,:,::-1] / 255.0
        out_path = output_path / img_name
        plot_all(img, current_gts, our_dets, base_dets, calib, str(out_path))

        print()
        counter += 1
        scores[frame_id] = improvement_counter
        
print(f"\n\nFound {counter} candidates overall.")
print("Top 50:")
string = "\n".join([str(it[0]) +": " + str(it[1]) for it in list(reversed(sorted(scores.items(), key=operator.itemgetter(1))))[:50]])
print(string)
print("Green: Ground truth")
print("Red: Baseline")
print("Blue: Ours")

with open(output_path / "top50.txt", "w") as file:
    file.write(string)