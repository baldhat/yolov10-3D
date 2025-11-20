import os

import cv2
import torch
import pathlib
from ultralytics.data.datasets.decode_helper import  *
from copy import deepcopy
# from ultralytics.data.datasets.kitti_eval import eval_from_scrach
import subprocess
from ultralytics.data.datasets.kitti_utils import get_objects_from_label, Calibration
from pathlib import Path
from ultralytics.utils.metrics import bbox_iou
import numpy as np
from qualitative_candidates import plot_bev

import torch.utils.data as data
from PIL import Image

from ultralytics.utils.ops import  xyxy2xywh, xywh2xyxy

img_path = "/storage/group/deepscenario/KITTI/training/image_2"
label_dir = "/storage/group/deepscenario/KITTI/training/label_2"
calib_dir = "/storage/group/deepscenario/KITTI/training/calib"
out_path = "/storage/user/mejo/tmp/"

def get_label(idx):
    label_file = os.path.join(label_dir, '%06d.txt' % idx)
    assert os.path.exists(label_file)
    return get_objects_from_label(label_file)

def get_calib(idx):
    calib_file = os.path.join(calib_dir, '%06d.txt' % idx)
    assert os.path.exists(calib_file)
    return Calibration(calib_file)

np.random.seed(1234)

counter = 0
for i in [7]: #range(7, 7000):
    option = False
    for j in [1251]: #range(7000):
        if i == j:
            continue
        img_file = '%06d.png' % i
        img_file2 = '%06d.png' % j
        
        labels1 = get_label(i)
        labels2 = get_label(j)
        calib = get_calib(i)
        
        option = False
        for obj1 in labels1:
            for obj2 in labels2:
                box1 = torch.tensor(obj1.box2d).unsqueeze(0)
                box2 = torch.tensor(obj2.box2d).unsqueeze(0)
                iou = bbox_iou(box1, box2, xywh=False)
                if iou[0].item() > 0.9 and box1[0, 0] > 50 and box1[0, 0] < 900:
                    print(box1)
                    print(box2)
                    option = True 
                    break
        
        if option:
            img1 = cv2.imread(img_path + "/" + img_file)
            img2 = cv2.imread(img_path + "/" + img_file2)
            img2 = Image.fromarray(img2).resize((img1.shape[1], img1.shape[0]))
            img1 = Image.fromarray(img1) 
            img = Image.blend(img1, img2, 0.5)
            cv2.imwrite(f"/home/stud/mijo/dev/yolov10-3D/output/option{counter}.png", np.array(img))
            cv2.imwrite(f"/home/stud/mijo/dev/yolov10-3D/output/raw0_{counter}.png", np.array(img1))
            cv2.imwrite(f"/home/stud/mijo/dev/yolov10-3D/output/raw1_{counter}.png", np.array(img2))
            
            labels1.extend(labels2)
            print(labels1[0].pos)
            d2 = []
            for gt in labels1:
                pred = deepcopy(gt)
                offset = np.random.randn(3) / 2
                offset[2] *= pred.pos[2] / 20
                pred.pos += offset
                pred.l += np.random.randn(1)[0] / 5
                pred.w += np.random.randn(1)[0] / 5
                #pred.ry += np.random.rand(1) * 0.2
                d2.append(pred)
            # print(d2[0].pos)
            d3 = []
            for gt in labels1:
                pred = deepcopy(gt)
                offset = np.random.randn(3) / 2
                offset[2] *= pred.pos[2] / 20
                pred.pos += offset
                pred.l += np.random.randn(1)[0] / 5
                pred.w += np.random.randn(1)[0] / 5
                #pred.ry += np.random.rand(1) * 0.2
                d3.append(pred)
            # print(d3[0].pos)
            
            d3[7], d3[1] = deepcopy(d3[1]), deepcopy(d3[7])
            d2[7].pos += np.random.randn(3) / 2
            d2[1].pos += np.random.randn(3) / 2
            # print(d2[0].occlusion)
            # print(d2[7].occlusion)
            # print(d2[0].trucation)
            # print(d2[7].trucation)
            
            plot_bev(labels1, d3, d2, f"/home/stud/mijo/dev/yolov10-3D/output/bev{counter}.svg", np.rad2deg(2*np.arctan2(1242, 2* calib.fu)))
            
            print(f"Found:{counter} : {i:06d} - {j:06d}")
            counter += 1
