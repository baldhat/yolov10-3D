from pathlib import Path
import numpy as np
import torch
import cv2 as cv
import os
import math
import operator

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

from ultralytics.data.datasets.kitti_utils import Object3d, Calibration, affine_transform
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.plotting import KITTIVisualizer, VisObject3D
from scipy.spatial.transform import Rotation

from ultralytics.utils.ops import  xyxy2xywh, xywh2xyxy
from scipy.optimize import linear_sum_assignment


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
    pos_errors = [np.linalg.norm(gt.pos[-1] - det.location[-1]) for gt,det in zip(gts, dets)]
    return pos_errors

def equals(gt1: Object3d, gt2: Object3d):
    return gt1.line_index == gt2.line_index

def load_calib(path):
    return Calibration(str(path))


if __name__=='__main__':
    test_plot = False
    
    val_files = Path("/storage/user/mijo/mixup_kitti/ImageSets/val.txt")
    gt_path = Path("/storage/user/mijo/mixup_kitti//training/label_2/")

    import sys
    if len(sys.argv) >= 2:
        print(sys.argv)
        base_path = Path(sys.argv[1])
        ours_path = Path(sys.argv[2])
        ours_name = str(ours_path).split("/")[-1]
        ours_name = str(base_path).split("/")[-1]
    else:
        #base_path = Path("/storage/user/mejo/_archiv_paper/2026_CVPR_LeAD-M3D/von_johannes/yolov10-3D_baseline_b")
        base_path = Path("/home/stud/mijo/dev/yolov10-3D/runs/detect/val-baseline2")
        ours_path = Path("/home/stud/mijo/dev/yolov10-3D/runs/detect/val-ours")


    counter = 0

    our_pos_errors = []
    base_pos_errors =  []

    for ci, fn in tqdm(enumerate(open(val_files, "r").readlines())):
        if ci >= 5000:
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

        
        # calculate position and rotation errors
        base_err = calculate_errors(base_gts, base_dets)
        our_err = calculate_errors(our_gts, our_dets)
        
        base_pos_errors.extend(base_err)
        our_pos_errors.extend(our_err)
    
    print("Mean Depth error:")
    print("base:", np.array(base_pos_errors).mean())
    print("ours:", np.array(our_pos_errors).mean())
    print("Median Depth error:")
    print("base:", torch.median(torch.tensor(np.array(base_pos_errors))).item())
    print("ours:", torch.median(torch.tensor(np.array(our_pos_errors))).item())
    print("5-quantile Depth error:")
    print("base:", torch.quantile(torch.tensor(np.array(base_pos_errors)), q=0.05).item())
    print("ours:", torch.quantile(torch.tensor(np.array(our_pos_errors)), q=0.05).item())
    print("95-quantile Depth error:")
    print("base:", torch.quantile(torch.tensor(np.array(base_pos_errors)), q=0.95).item())
    print("ours:", torch.quantile(torch.tensor(np.array(our_pos_errors)), q=0.95).item())
        
        
        
            