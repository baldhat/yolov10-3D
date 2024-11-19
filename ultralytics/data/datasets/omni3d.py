# Ultralytics YOLO 🚀, AGPL-3.0 license
import os

import torch.utils.data as data
from PIL import Image
import subprocess
from pathlib import Path

from scipy.spatial.transform import Rotation

from ultralytics.utils.ops import *

from collections import defaultdict
import json

from ultralytics.data.datasets.kitti_utils import get_objects_from_dict, Calibration, get_affine_transform, affine_transform


class Omni3Dataset(data.Dataset):
    def __init__(self, filepath, mode, args):
        self.args = args
        self.path = "/".join(filepath.split("/")[:-1])
        self.split = mode
        self.mode = mode
        self.class_name = ['Car', 'Pedestrian', 'Cyclist']
        self.writelist = ['Car', 'Pedestrian', 'Cyclist']
        self.resolution = np.array([960, 640])  # W * H
        self.max_objs = 50
        self.use_camera_dis = False

        print("Loading Omni3D Dataset...")
        self.raw_split = json.load(open(filepath, 'r'))
        if args.overfit:
            self.raw_split["images"] = [image for image in self.raw_split["images"] if image["id"] < 703600]
            self.raw_split["annotations"] = [anns for anns in self.raw_split["annotations"] if anns["image_id"] < 703600]

        self.imgs = {img['id']: img for img in sorted(self.raw_split['images'], key=lambda img: img['id'])}
        self.idx_to_img_id = {idx: img_id for idx, img_id in enumerate(self.imgs)}

        self.cls2train_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        self.train_id2cls = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

        self.data_cls2data_id = {value["name"].title(): value["id"] for value in self.raw_split["categories"]}
        self.data_id2data_cls = {cls_id: cls_name for cls_name, cls_id in self.data_cls2data_id.items()}
        self.cls2eval_id = {"unknown": 0, "Car": 1, "Pedestrian": 2, "Sign": 3, "Cyclist": 4}

        self.anns_by_img = defaultdict(list)
        for ii, ann in enumerate(self.raw_split['annotations']):
            ann["category"] = self.data_id2data_cls[ann["category_id"]]
            self.anns_by_img[ann['image_id']].append(ann)

        self.labels = self.get_labels()

        ##h,w,l
        self.cls_mean_size = np.array([
            [1.52563191462, 1.62856739989, 3.88311640418],
            [1.76255119, 0.66068622, 0.84422524],
            [1.73698127, 0.59706367, 1.76282397]])

        # data augmentation configuration
        self.data_augmentation = True if self.mode in ['train', 'trainval'] else False
        self.random_flip = args.fliplr
        self.random_crop = args.random_crop
        self.scale = args.scale
        self.min_scale = args.min_scale
        self.max_scale = args.max_scale
        self.shift = args.translate
        self.mixup = args.mixup
        self.max_depth_threshold = args.max_depth_threshold
        self.min_depth_thres = args.min_depth_threshold
        self.pred_rot_mat = args.pred_rot_mat
        self.load_depth_maps = False
        assert(self.pred_rot_mat)

        self.left_multiply_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.right_multiply_matrix = np.array([[-1, 0, 0], [0, 1, 0.], [0, 0, -1]])

        print("Finished loading!")

    def get_image(self, idx):
        img_file = os.path.join(self.path, self.imgs[idx]["file_path"].replace("waymo/images/", ""))
        if not os.path.exists(img_file):
            print(f"Missing segment: {img_file}")
        return Image.open(img_file).convert("RGB")

    def get_label(self, idx):
        return get_objects_from_dict(self.anns_by_img[idx])

    def get_labels(self):
        labels = self.anns_by_img
        labels = [item for sublist in labels.values() for item in sublist]
        labels = [item for item in labels if item["category"] in self.writelist]
        return labels

    def get_calib(self, idx):
        calib = np.array(self.imgs[idx]["K"])
        calib = np.hstack((calib, np.zeros((3, 1))))
        return Calibration({"P2": calib, 'R0': 0, "Tr_velo2cam": np.ones_like(calib)})

    def get_im_files(self):
        return self.imgs.values()

    def __len__(self):
        return self.imgs.__len__()

    def __getitem__(self, item):
        index = int(self.idx_to_img_id[item])  # index mapping, get real data id
        ori_img =  self.get_image(index)
        img = ori_img
        img_size = np.array(ori_img.size)
        if self.split != 'test':
            dst_W, dst_H = img_size

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size = img_size
        random_crop_flag, random_flip_flag = False, False
        random_mix_flag = False
        calib = self.get_calib(index)
        scale = 1
        shift = np.array([0, 0])

        if self.data_augmentation:
            if np.random.random() < 0.5 and self.mixup:
                random_mix_flag = True

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random() < self.random_crop:
                random_crop_flag = True
                # scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                scale_variance = (self.max_scale - self.min_scale) / 2
                scale_mean = (self.max_scale + self.min_scale) / 2
                scale = np.clip(np.random.randn() * scale_variance + scale_mean, self.min_scale, self.max_scale)
                crop_size = img_size * scale
                shift[0] = img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                shift[1] = img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[0] += shift[0]
                center[1] += shift[1]

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = np.random.randint(self.__len__())
                random_index = int(self.idx_to_img_id[random_index])
                calib_temp = self.get_calib(random_index)

                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs:
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # C * H * W

        ratio_pad = np.array([self.resolution / img_size, np.array([0, 0])])

        #  ============================   get labels   ==============================
        gt_boxes_2d = []
        gt_cls = []
        gt_center_2d = []
        gt_center_3d = []
        gt_size_2d = []
        gt_size_3d = []
        gt_depth = []
        gt_heading_bin = []
        gt_heading_res = []
        gt_rot_mat = []

        if self.split != 'test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d # xyxy
                    object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.pos[0] *= -1
                    object.rot_mat = self.left_multiply_matrix @ object.rot_mat @ self.right_multiply_matrix

            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            for i in range(object_num):
                valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat \
                    = self.load_object(objects[i], scale, trans, calib, shift, ratio_pad)
                if valid:
                    gt_boxes_2d.append(_box)
                    gt_cls.append(_cls)
                    gt_center_2d.append(_center2d)
                    gt_center_3d.append(_center3d)
                    gt_size_2d.append(_size2d)
                    gt_size_3d.append(_size3d)
                    gt_depth.append(_depth)
                    gt_heading_bin.append(_head_bin)
                    gt_heading_res.append(_head_res)
                    gt_rot_mat.append(_rot_mat)

            if random_mix_flag == True:
                objects = self.get_label(random_index)
                # data augmentation for labels
                if random_flip_flag:
                    for object in objects:
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                        object.pos[0] *= -1
                        object.rot_mat = self.left_multiply_matrix @ object.rot_mat @ self.right_multiply_matrix
                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (
                        self.max_objs - object_num)
                for i in range(object_num_temp):
                    valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat \
                        = self.load_object(objects[i], scale, trans, calib, shift, ratio_pad)
                    if valid:
                        gt_boxes_2d.append(_box)
                        gt_cls.append(_cls)
                        gt_center_2d.append(_center2d)
                        gt_center_3d.append(_center3d)
                        gt_size_2d.append(_size2d)
                        gt_size_3d.append(_size3d)
                        gt_depth.append(_depth)
                        gt_heading_bin.append(_head_bin)
                        gt_heading_res.append(_head_res)
                        gt_rot_mat.append(_rot_mat)

        inputs = torch.tensor(img)
        info = {'img_id': index,
                'img_size': img_size,
                'trans_inv': trans_inv}

        if len(gt_boxes_2d) > 0:
            # We need xywh in [0, 1]
            bboxes = torch.clip(torch.tensor(np.array(gt_boxes_2d) / self.resolution[[0, 1, 0, 1]]), 0, 1)
        else:
            bboxes = torch.empty(0)
        calib_tensor = torch.tensor(np.array([calib.cu * ratio_pad[0, 0], calib.cv * ratio_pad[0, 1],
                                              calib.fu * ratio_pad[0, 0], calib.fv * ratio_pad[0, 1],
                                              calib.tx * ratio_pad[0, 0], calib.ty * ratio_pad[0, 1]]))

        data = {
            "img": inputs,
            "ori_img": ori_img,
            "calib": calib_tensor,
            "info": info,
            "cls": torch.tensor(np.array(gt_cls)),
            "bboxes": bboxes,
            "batch_idx": torch.zeros(len(gt_boxes_2d)),  # Used during collate_fn
            "im_file": '%06d.txt' % info["img_id"],
            "ori_shape": info["img_size"][::-1],  # this one is (height, width)
            "ratio_pad": torch.tensor(ratio_pad),
            "center_2d": torch.tensor(np.array(gt_center_2d)),
            "center_3d": torch.tensor(np.array(gt_center_3d)),
            "size_2d": torch.tensor(np.array(gt_size_2d)),
            "size_3d": torch.tensor(np.array(gt_size_3d)),
            "depth": torch.tensor(np.array(gt_depth)),
            "mean_sizes": torch.tensor(self.cls_mean_size),
            "heading_bin": torch.tensor(np.array(gt_heading_bin)),
            "heading_res": torch.tensor(np.array(gt_heading_res)),
            "mixed": torch.tensor(np.array(random_mix_flag, dtype=np.uint8)),
            "rot_mat": torch.tensor(gt_rot_mat),
            "shift": torch.tensor(shift)
        }
        return data

    def load_object(self, object_, scale, trans, calib, shift, ratio_pad):
        valid = False
        _box = 0
        _cls = 0
        _center2d = 0
        _center3d = 0
        _size2d = 0
        _size3d = 0
        _depth = 0
        _head_bin = 0
        _head_res = 0
        _rot_mat = 0

        if ((object_.cls_type not in self.writelist)
                or (object_.behind_camera or (object_.pos[-1] * scale < self.min_depth_thres))
                or not object_.valid3D or (object_.num_lidar != -1 and object_.num_lidar < 5) or object_.depth_error >= 0.5
                or (object_.truncation >= 0.75 or (object_.visibility <= 0.25 and object_.visibility != -1))):
            return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat

        # process 3d bbox & get 3d center
        center_3d = convert_location_ground2gravity(
                    ground_center=torch.tensor(object_.pos)[None].float(),
                    egoc_rot_matrix=torch.tensor(object_.rot_mat).unsqueeze(0).float(),
                    dim=torch.tensor(np.array([object_.h, object_.w, object_.l], dtype=np.float32)).unsqueeze(0).float())[0].numpy()
        r_center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
        center_3d, _ = calib.rect_to_img(r_center_3d)  # project 3D center to image plane
        center_3d = center_3d[0]  # shape adjustment
        center_3d = affine_transform(center_3d.reshape(-1), trans)
        #center_3d -= trans[[0, 1], [2, 2]]
        _center3d = center_3d.copy()

        # process 2d bbox & get 2d center
        bbox_2d = object_.box2d
        # add affine transformation for 2d boxes.
        bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
        bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

        _box = np.copy(bbox_2d)
        _box = xyxy2xywh(_box)
        _size2d = _box[2:]
        _center2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                             dtype=np.float32)  # W * H

        # generate the center of gaussian heatmap [optional: 3d center or 2d center]
        center_heatmap = center_3d.astype(np.int32)
        if (center_heatmap[0] < 0 or center_heatmap[0] >= self.resolution[0]
                or center_heatmap[1] < 0 or center_heatmap[1] >= self.resolution[1]):
            return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat

        # encoding depth
        depth = r_center_3d[0, -1]
        depth *= scale
        if depth > self.max_depth_threshold:
            return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat

        _cls = [self.cls2train_id[object_.cls_type]]

        in_pos = np.array([_center3d[0] + shift[0]*trans[0, 0], _center3d[1] + shift[1]*trans[1, 1]])
        _rot_mat = egoc_to_alloc_rot_matrix_torch(amodal_center=torch.from_numpy(in_pos).unsqueeze(0).float(),
                                                  egoc_rot_matrix=torch.from_numpy(object_.rot_mat).unsqueeze(0).float(),
                                                  calib=torch.from_numpy(calib.P2*ratio_pad[0, 0]).unsqueeze(0).float())[0].numpy().reshape(9)

        _size3d = (np.array([object_.h, object_.w, object_.l], dtype=np.float32)
                   - self.cls_mean_size[self.cls2train_id[object_.cls_type]])

        if self.use_camera_dis:
            r_center_3d *= scale
            dep = np.linalg.norm(r_center_3d)
            _depth = dep
        else:
            _depth = depth
        valid = True
        return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat

    def get_preds_and_gts(self, results):
        pred_annos = {"bbox": [], "type": [], 'frame_id': [], "score": []}
        gt_annos = {"bbox": [], "type": [], 'frame_id': [], "score": [], "diff": []}

        for (key, value) in results.items():
            frame_id = int(key.split(".")[0])

            for pred in results[key]:
                cls = self.cls2eval_id[self.train_id2cls[int(pred[0])]]
                egoc_rot_matrix = np.array(pred[1:10]).reshape(3, 3)
                dim = [pred[16], pred[15], pred[14]] # h,w,l -> l,w,h
                location = pred[17:20]
                ry = [-Rotation.from_matrix(egoc_rot_matrix).as_euler("xyz")[1]]
                score = float(pred[-1])
                pred_annos["bbox"].append(location + dim + ry)
                pred_annos["type"].append(cls)
                pred_annos["frame_id"].append(frame_id)
                pred_annos["score"].append(score)

            for gt in self.anns_by_img[frame_id]:
                cls = self.data_cls2data_id[gt["category"]]
                dim = gt["dimensions"]
                location = gt["center_cam"]
                ry = [-Rotation.from_matrix(gt['R_cam']).as_euler('xyz')[1]]
                score = "1.0"
                gt_annos["bbox"].append(location + [dim[2], dim[0], dim[1]] + ry) # h,w,l -> l,w,h
                #gt_annos["bbox"].append(location + dim + ry)  FIXME
                gt_annos["type"].append(cls)
                gt_annos["frame_id"].append(frame_id)
                gt_annos["score"].append(score)
                gt_annos['diff'].append(2)

        return pred_annos, gt_annos

    def get_stats(self, results, save_dir):
        pred_annos, gt_annos = self.get_preds_and_gts(results)

        file_path = os.path.join(save_dir, "eval_results.json")
        with open(file_path, "w") as f:
            json.dump({
                "pred": pred_annos,
                "gt": gt_annos
            }, f)

        python = os.path.join(Path.home(), "anaconda3/envs/py36_waymo_tf/bin/python")
        if not os.path.exists(python):
            python = os.path.join(Path.home(), "miniconda3/envs/py36_waymo_tf/bin/python")
        command = f"{python} -u ultralytics/data/datasets/waymo_eval.py --iou 0.7 --pred {file_path}"
        lines = subprocess.check_output(command, shell= True, text= True, env={})

        print(lines)
        metric3d = float(lines.split("\n")[4].split("|")[2].strip().split(" ")[0])
        return metric3d

    def decode_preds_eval(self, preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=True,
                          threshold=0.001, ground_center=False):
        return self.decode_preds(preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=undo_augment,
                                 threshold=threshold, ground_center=False)

    def decode_batch_eval(self, batch, calibs, undo_augment=True):
        return self.decode_batch(batch, calibs, undo_augment=undo_augment)

    def decode_batch(self, batch, calibs, undo_augment=True):
        results = {}
        for i in range(batch["img"].shape[0]):
            targets = []
            mask = batch["batch_idx"] == i
            num_targets = mask.sum()
            for j in range(num_targets):
                cls_id = batch["cls"][mask][j].item()
                bbox = batch["bboxes"][mask][j].cpu().numpy()
                bbox = (xywh2xyxy(bbox) * batch["ori_shape"][i][[1, 0, 1, 0]]).tolist()

                dimensions = batch["size_3d"][mask][j].cpu().numpy()
                dimensions += self.cls_mean_size[int(cls_id)]

                depth = batch["depth"][mask][j].cpu().numpy()

                if undo_augment:
                    x3d = batch["center_3d"][mask][j, 0].cpu().numpy()
                    y3d = batch["center_3d"][mask][j, 1].cpu().numpy()
                    c3d = affine_transform(np.array([x3d, y3d]), np.array(batch["info"][i]["trans_inv"]))
                    if self.use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                else:
                    x3d = batch["center_3d"][mask][j, 0].cpu().numpy() / batch["ratio_pad"][i][0, 0]
                    y3d = batch["center_3d"][mask][j, 1].cpu().numpy() / batch["ratio_pad"][i][0, 1]
                    if self.use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(x3d, y3d, depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)

                egoc_rot_mat = alloc_to_egoc_rot_matrix_torch(
                    amodal_center=torch.tensor(np.array([x3d, y3d])).cpu().unsqueeze(0),
                    alloc_rot_matrix=batch["rot_mat"][mask][j].cpu().unsqueeze(0).reshape(1, 3, 3),
                    calib=torch.tensor(calibs[i].P2).unsqueeze(0).cpu()
                )[0].numpy()

                locations = convert_location_gravity2ground(
                    gravity_center=torch.tensor(locations)[None].float().cpu(),
                    egoc_rot_matrix=torch.tensor(egoc_rot_mat)[None].float().cpu(),
                    dim=torch.tensor(dimensions).unsqueeze(0).float().cpu())[0].numpy()

                score = 1

                targets.append([cls_id] + egoc_rot_mat.ravel().tolist() + bbox + dimensions.tolist() + locations.tolist() + [score])

            results[batch["im_file"][i]] = targets
        return results

    def decode_preds(self, preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=True,
                     threshold=0.001, ground_center=True):
        preds = preds.detach().cpu()
        bboxes, pred_center3d, pred_s3d, pred_rot_mat, pred_dep, pred_dep_un, scores, labels = preds.split(
            (4, 2, 3, 9, 1, 1, 1, 1), dim=-1)

        scores = scores.sigmoid()

        results = {}
        for i, img in enumerate(preds):
            targets = []
            for j, pred in enumerate(img):
                cls_id = labels[i, j].item()
                dimensions = pred_s3d[i, j].numpy()
                dimensions += self.cls_mean_size[int(cls_id)]
                bbox = bboxes[i, j].cpu().numpy()
                bbox = (xywh2xyxy(bbox) / ratio_pad[i][0, [1, 0, 1, 0]]).tolist()

                depth = pred_dep[i, j].numpy()
                sigma = torch.exp(-pred_dep_un[i, j]).item()

                if undo_augment:
                    x3d = pred_center3d[i, j, 0].numpy()
                    y3d = pred_center3d[i, j, 1].numpy()
                    c3d = affine_transform(np.array([x3d, y3d]), np.array(inv_trans[i]))
                    if self.use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                else:
                    x3d = pred_center3d[i, j, 0].numpy() / ratio_pad[i][0, 0]
                    y3d = pred_center3d[i, j, 1].numpy() / ratio_pad[i][0, 1]
                    if self.use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(x3d, y3d, depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)

                egoc_rot_mat = alloc_to_egoc_rot_matrix_torch(
                    amodal_center=torch.tensor(np.array([x3d, y3d])).unsqueeze(0).cpu(),
                    alloc_rot_matrix=pred_rot_mat[i, j].unsqueeze(0).reshape(1, 3, 3).cpu(),
                    calib=torch.tensor(calibs[i].P2).unsqueeze(0).cpu()
                )[0].numpy()

                if ground_center:
                    locations = convert_location_gravity2ground(
                        gravity_center=torch.tensor(locations)[None].float().cpu(),
                        egoc_rot_matrix=torch.tensor(egoc_rot_mat)[None].float().cpu(),
                        dim=torch.tensor(dimensions).unsqueeze(0).float().cpu()
                    )[0].numpy()

                score = scores[i, j].item() * sigma
                if score < threshold:
                    continue

                targets.append([cls_id] + egoc_rot_mat.ravel().tolist() + bbox + dimensions.tolist() + locations.tolist() + [score])

            results[im_files[i]] = targets
        return results

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in ["img", "coord_range", "ratio_pad", "calib", "mixed", "shift"]:
                value = torch.stack(value, 0)
            if k in ["bboxes", "cls", "depth", "center_3d", "center_2d", "size_2d", "heading_bin",
                     "heading_res", "size_3d", "rot_mat"]:
                value = torch.cat(value, 0)
            if k not in ["mean_sizes"]:
                new_batch[k] = value
            else:
                new_batch[k] = batch[0][k]
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for j in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][j] += j  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    @staticmethod
    def sort_by_id(categories):
        return sorted(categories, key=lambda cat: cat['id'])
