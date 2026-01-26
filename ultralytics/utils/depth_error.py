from pathlib import Path
import numpy as np
import torch
import cv2 as cv
import os
import math
import operator

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

def load_labels(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        return [Object3d(it, idx=i) for i, it in enumerate(lines)]
    
def load_dets(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        return [Detection3d(it) for it in lines]
    
def filter_(dets):
    return [det for det in dets if det.score > 0.1 and det.classname in ["Car", "Pedestrian", "Cyclist", "Van"]]


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


def filter_gts(dets: [Object3d], calib):
    out = []
    for gt in dets:
        if gt.level_str == 'UnKnown' or np.linalg.norm(gt.pos) > 60:
            continue
        if gt.trucation > 0.5 or gt.occlusion > 2:
            continue
        if gt.cls_type not in ["Car", "Pedestrian", "Cyclist", "Van"]:
            continue
        if not center_inside_image(gt, calib):
            continue
        out.append(gt)
    return out

def associate(gts: [Object3d], dets: [Detection3d]):
    if len(dets) == 0:
        return [], [], []
    iou = box_iou(torch.tensor(np.array([it.box2d for it in gts])), torch.tensor(np.array([it.bbox for it in dets])))
    false_positives = list(np.where(np.all(iou.cpu().numpy() == 0, axis=0))[0])
    row_ind, col_ind = linear_sum_assignment(iou.cpu().detach().numpy(), maximize=True)
    matched_gts = [gts[ind_r] for ind_r, ind_c in zip(row_ind, col_ind)]
    matched_dets = [dets[ind_c] for ind_r, ind_c in zip(row_ind, col_ind)]
    return matched_gts, matched_dets, [dets[fp] for fp in false_positives]

def calculate_errors(gts: [Object3d], dets: [Detection3d]):
    pos_errors = [np.linalg.norm(gt.pos - det.location) for gt,det in zip(gts, dets)]
    rot_errors = [np.abs((gt.ry - det.ry)%np.pi) for gt,det in zip(gts, dets)]
    return pos_errors, rot_errors

def equals(gt1: Object3d, gt2: Object3d):
    return gt1.line_index == gt2.line_index

def load_calib(path):
    return Calibration(str(path))

def load_image(path):
    return cv.imread(str(path))

def plot_labels(img, gts: [Object3d], calib, color):
    for object in gts:
        cls = object.cls_type
        bbox2d = object.box2d
        dimensions = np.array([object.l, object.w, object.h])
        translation = object.pos
        ry = object.ry
        egoc_rot_matrix = plotter.get_egoc_rot_matrix(ry)

        plotter.plot_3d_obj(img,
                            VisObject3D(translation, Rotation.from_matrix(egoc_rot_matrix).as_rotvec(),
                                        dimensions, bbox2d, cls),
                            calib.P2, color=gt_color)

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
    plotter.plot_3d_obj(img, objects,calib.P2, [colors(i % 10) for i,_ in enumerate(objects)])

def plot_bev(gts, base_dets, our_dets, filename, fov=60):
    print(fov)
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

    for j, object in enumerate(gts):            
        dimensions = np.array([object.l, object.w])
        translation = object.pos[[0, 2]]
        ry = -object.ry

        corners = get_rotated_rectangle_points(translation, dimensions, ry * 180 / np.pi)
        art = ax.add_artist(Polygon(corners, closed=True, fill=False, edgecolor=gt_color, facecolor=gt_color, zorder=3, linewidth=5))
        if j == 0:
            art.set_label("Ground Truth")
        
    for j, object in enumerate(base_dets):
        dimensions = object.dimensions[::-1][:2]
        translation = object.location[[0, 2]]
        ry = -object.ry

        corners = get_rotated_rectangle_points(translation, dimensions, ry * 180 / np.pi)
        art = ax.add_artist(Polygon(corners, closed=True, fill=False, edgecolor=base_color, facecolor=base_color, zorder=3, linewidth=5))
        if j == 0:
            art.set_label("Baseline")
        
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
    print(filename)

def plot_all(img, gts, our_dets, base_dets, calib, out_path):
    base_img = img.copy()
    our_img = img.copy()
        
    # plot_labels(our_img, gts, calib, color="g")
    plot_dets(our_img, our_dets, calib)
    our_name = out_path.replace(".png", "_ours.png")
    cv.imwrite(our_name, (our_img*255.0).astype(np.uint8))
    print(our_name)
    
    # plot_labels(base_img, gts, calib, color="g")
    # plot_dets(base_img, base_dets, calib, color="r")
    # base_name = out_path.replace(".png", "_base.png")
    # cv.imwrite(base_name, (base_img*255.0).astype(np.uint8))
    # print(base_name)
    
    plot_bev(gts, base_dets, our_dets, out_path.replace(".png", "_bev.svg"), np.rad2deg(2*np.arctan2(base_img.shape[1], 2* calib.fu)))
    
if __name__=='__main__':
    test_plot = False
    
    val_files = Path("/storage/group/deepscenario/KITTI/ImageSets/val.txt")

    import sys
    if len(sys.argv) >= 2:
        print(sys.argv)
        base_path = Path(sys.argv[1])
        ours_path = Path(sys.argv[2])
        ours_name = str(ours_path).split("/")[-1]
        ours_name = str(base_path).split("/")[-1]
    else:
        #base_name = "yolov10-3D_kitti_baseline_x_117"
        base_name = "MonoLSS_predictions_600_epochs_kitti_val"
        ours_name = "val"
        base_path = Path("/storage/group/deepscenario/for_jonathan/" + base_name)
        ours_path = Path("/home/stud/mijo/dev/yolov10-3D/runs/detect/" + ours_name)

    output_path = Path("/storage/group/deepscenario/jonathan_for_johannes/") / ours_name
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    gt_path = Path("/storage/group/deepscenario/KITTI/training/label_2/")

    counter = 0

    scores = {}

    for ci, fn in enumerate(open(val_files, "r").readlines()):
        if test_plot and ci > 3:
            break
        filename = fn.strip() + ".txt"
        plot = False
        # load dets and gts
        if os.path.exists(base_path / filename):
            base_dets = load_dets(base_path / filename)
        else:
            base_dets = load_dets(base_path / "preds" / filename)
        our_dets = load_dets(ours_path / "preds" / filename)
        gts = load_labels(gt_path / filename)

        calib = load_calib(gt_path / ".." / "calib" / filename)
        gts = filter_gts(gts, calib)

        if len(gts) == 0:
            continue
        
        # filter dets by score and class
        base_dets_ = filter_(base_dets)
        our_dets_ = filter_(our_dets)
        if len(our_dets_) == 0:
            continue
        
        # associate dets to gts
        base_gts, base_dets, base_false_positives = associate(gts, base_dets_)
        our_gts, our_dets, our_false_positives = associate(gts, our_dets_)
        
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
            for i, base_gt in enumerate(base_gts):
                found = True
                if not equals(base_gt, our_gt):
                    continue
                
                diff = base_pos_errors[i] - our_pos_errors[j]
                if diff > 0.7 and diff < 15:
                    #print(f"Better Location! Base: {base_dets[i].location}, Ours: {our_dets[j].location}")
                    improvement_counter += 1 #math.ceil(base_pos_errors[i] - our_pos_errors[j] - 5)
                    
                if np.abs(base_rot_errors[i] - our_rot_errors[j]) > 1:
                    #print(f"Better Rotation! Base: {base_dets[i].ry}, Ours: {our_dets[j].ry}")
                    #plot = True
                    pass
            if not found:
                pass
                #print("We detected more objects")
                improvement_counter += 1
                    
        if improvement_counter >= 2 or test_plot:
            print(filename)
            img_name = filename.replace("txt", "png")
            img = load_image(gt_path / ".." / "image_2" / img_name).astype(np.float32) / 255.0
            out_path = output_path / img_name
            plot_all(img, gts, our_dets, base_dets, calib, str(out_path))
            print()
            counter += 1
            scores[filename] = improvement_counter
            
    print(f"\n\nFound {counter} candidates overall.")
    print("Top Ten:")
    print("\n".join([it[0] +": " + str(it[1]) for it in list(reversed(sorted(scores.items(), key=operator.itemgetter(1))))[:10]]))