import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import cv2

def plot_bev(target, img, color=(0, 255, 0)):
    height, width, length, x, y, z, roty = target
    dimensions = np.array([length, width]) * SCALE
    translation = np.array((x, z)) * SCALE
    translation[1] *= -1
    translation += R

    bev = np.concatenate((translation, dimensions, np.expand_dims(roty, 0)))
    box = cv2.boxPoints((bev[:2], bev[2:4], bev[4] * 180 / np.pi)).astype(np.int32)
    img = cv2.drawContours(img, [box], -1, color, thickness=-1, lineType=cv2.LINE_AA)



pred_directory = Path("/home/stud/mijo/dev/yolov10-3D/runs/detect/val2/preds")
target_directory = Path("/home/stud/mijo/storage/group/deepscenario/rope3d/val/label_2/")

file_name = "149116_sj8fas6n151d20211125air_420_1637222649_1637225138_357_obstacle.txt"
pred_file = pred_directory / file_name

MAX_DIST = 120
SCALE = 10

# Create BEV Space
R = (MAX_DIST * SCALE)
space = np.zeros((R * 2, R * 2, 3), dtype=np.uint8)

for theta in np.linspace(0, np.pi, 7):
    space = cv2.line(space, pt1=(int(R - R * np.cos(theta)), int(R - R * np.sin(theta))), pt2=(R, R),
                        color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

for radius in np.linspace(0, R, 5):
    if radius == 0:
        continue

    space = cv2.circle(space, center=(R, R), radius=int(radius), color=(255, 255, 255), thickness=2,
                        lineType=cv2.LINE_AA)
space = space[:R, :, :]

target_file = target_directory / file_name
with open(target_file) as tf:
    targets = tf.readlines()
    for target in targets:
        cls, _, _, alpha, bbox_1, bbox_2, bbox3, bbox4, dim1, dim2, dim3, x, y, z, roty = target.split(" ")
        if cls != "car":
            continue
        target = (float(dim1), float(dim2), float(dim3), float(x), float(y), float(z), float(roty))
        plot_bev(target, space)
        
with open(pred_file) as tf:
    preds = tf.readlines()
    for pred in preds:
        cls, _, _, alpha, bbox_1, bbox_2, bbox3, bbox4, dim1, dim2, dim3, x, y, z, roty, score = pred.split(" ")
        if cls != "car" or float(score) < 0.1:
            continue
        pred = (float(dim1), float(dim2), float(dim3), float(x), float(y), float(z), float(roty))
        plot_bev(pred, space, color=(0, 0, 255))

cv2.imwrite("/home/stud/mijo/tmp/bev2.png", space)