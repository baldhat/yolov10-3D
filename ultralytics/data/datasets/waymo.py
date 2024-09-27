# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
from ultralytics.data.datasets.decode_helper import  *
from ultralytics.utils.keypoint_utils import get_object_keypoints

import torch.utils.data as data
from PIL import Image
import subprocess
import torch
from pathlib import Path

from collections import defaultdict
import json

from ultralytics.data.utils import angle2class
from ultralytics.data.datasets.kitti_utils import get_objects_from_dict, Calibration, get_affine_transform, affine_transform

from ultralytics.utils.ops import xyxy2xywh, xywh2xyxy


class WaymoDataset(data.Dataset):
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

        self.raw_split = json.load(open(filepath, 'r'))
        if args.overfit:
            self.raw_split["images"] = [image for image in self.raw_split["images"] if image["id"] < 50]
            self.raw_split["annotations"] = [anns for anns in self.raw_split["annotations"] if anns["image_id"] < 50]

        self.imgs = {img['id']: img for img in sorted(self.raw_split['images'], key=lambda img: img['id'])}
        self.idx_to_img_id = {idx: img_id for idx, img_id in enumerate(self.imgs)}

        self.cls2data_id = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
        self.cls2train_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        self.data_id2cls = {1: "Car", 2: "Pedestrian", 3: "Cyclist"}
        self.train_id2cls = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

        self.anns_by_img = defaultdict(list)
        for ii, ann in enumerate(self.raw_split['annotations']):
            ann['train_obj_id'] = ii
            ann["category"] = self.data_id2cls[ann["category_id"]]
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

    def get_image(self, idx):
        img_file = os.path.join(self.path, self.imgs[idx]["file_name"])
        if not os.path.exists(img_file):
            print(f"Missing segment: {img_file}")
        return Image.open(img_file).convert("RGB")

    def get_label(self, idx):
        return get_objects_from_dict(self.anns_by_img[idx])

    def get_labels(self):
        labels = self.anns_by_img
        labels = [item for sublist in labels.values() for item in sublist]
        labels = [item for item in labels if self.data_id2cls[item["category_id"]] in self.writelist]
        return labels

    def get_calib(self, idx):
        calib = self.imgs[idx]["calib"]
        return Calibration({"P2": np.array(calib), 'R0': 0, "Tr_velo2cam": np.ones_like(calib)})

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
                shift_0 = img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                shift_1 = img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                center[0] += shift_0
                center[1] += shift_1

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

        coord_range = np.array([center - crop_size / 2, center + crop_size / 2]).astype(np.float32)
        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # C * H * W

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

        if self.split != 'test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d
                    object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.ry = np.pi - object.ry
                    object.pos[0] *= -1
                    if object.ry > np.pi:  object.ry -= 2 * np.pi
                    if object.ry < -np.pi: object.ry += 2 * np.pi

            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            for i in range(object_num):
                valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res \
                    = self.load_object(objects[i], scale, trans, calib)
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

            if random_mix_flag == True:
                # if False:
                objects = self.get_label(random_index)
                # data augmentation for labels
                if random_flip_flag:
                    for object in objects:
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                        object.ry = np.pi - object.ry
                        object.pos[0] *= -1
                        if object.ry > np.pi:  object.ry -= 2 * np.pi
                        if object.ry < -np.pi: object.ry += 2 * np.pi
                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (
                        self.max_objs - object_num)
                for i in range(object_num_temp):
                    valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res = (
                        self.load_object(objects[i], scale, trans, calib))
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

        inputs = torch.tensor(img)
        info = {'img_id': index,
                'img_size': img_size,
                'trans_inv': trans_inv}

        if len(gt_boxes_2d) > 0:
            # We need xywh in [0, 1]
            bboxes = torch.clip(torch.tensor(np.array(gt_boxes_2d) / self.resolution[[0, 1, 0, 1]]), 0, 1)
        else:
            bboxes = torch.empty(0)
        ratio_pad = np.array([self.resolution / img_size, np.array([0, 0])])
        calib = torch.tensor(np.array([calib.cu * ratio_pad[0, 0], calib.cv * ratio_pad[0, 1],
                                       calib.fu * ratio_pad[0, 0], calib.fv * ratio_pad[0, 1],
                                       calib.tx * ratio_pad[0, 0], calib.ty * ratio_pad[0, 1]]))

        data = {
            "img": inputs,
            "ori_img": ori_img,
            "calib": calib,
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
        }
        return data

    def load_object(self, object_, scale, trans, calib):
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

        if ((object_.cls_type not in self.writelist)
                or (object_.level_str == 'UnKnown' or (object_.pos[-1] * scale < self.min_depth_thres))
                or (object_.cls_type == "Car" and object_.num_lidar <= 100)
                or (object_.cls_type != 'Car' and object_.num_lidar <= 50)):
            return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res

        # process 3d bbox & get 3d center
        center_3d = object_.pos - [0, object_.h / 2, 0]  # real 3D center in 3D space
        r_center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
        center_3d, _ = calib.rect_to_img(r_center_3d)  # project 3D center to image plane
        center_3d = center_3d[0]  # shape adjustment
        center_3d = affine_transform(center_3d.reshape(-1), trans)
        _center3d = center_3d.copy()

        # process 2d bbox & get 2d center
        bbox_2d = self.recompute_bbox_2d(r_center_3d, object_, object_.ry, calib)
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
            return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res

        # encoding depth
        depth = object_.pos[-1]
        depth *= scale
        if depth > self.max_depth_threshold:
            return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res

        _cls = [self.cls2train_id[object_.cls_type]]

        # encoding heading angle
        # heading_angle = objects[i].alpha
        heading_angle = calib.ry2alpha(object_.ry, (object_.box2d[0] + object_.box2d[2]) / 2)
        if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
        if heading_angle < -np.pi: heading_angle += 2 * np.pi
        heading_bin, heading_res = angle2class(heading_angle)
        _head_bin = heading_bin
        _head_res = heading_res

        _size3d = (np.array([object_.h, object_.w, object_.l], dtype=np.float32)
                   - self.cls_mean_size[self.cls2train_id[object_.cls_type]])

        if self.use_camera_dis:
            r_center_3d *= scale
            dep = np.linalg.norm(r_center_3d)
            _depth = dep
        else:
            _depth = depth
        valid = True
        return valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res

    def recompute_bbox_2d(self, center3d, object_, roty, calib):
        kpts_3d = get_object_keypoints(center3d, object_.dims, roty).squeeze(0).squeeze(0)
        kpts_2d = calib.rect_to_img(kpts_3d.numpy())[0]

        x0y0 = np.min(kpts_2d, axis=0)
        x1y1 = np.max(kpts_2d, axis=0)

        return np.concatenate((x0y0, x1y1))

    def get_preds_and_gts(self, results):
        pred_annos = {"bbox": [], "type": [], 'frame_id': [], "score": []}
        gt_annos = {"bbox": [], "type": [], 'frame_id': [], "score": [], "diff": []}

        for (key, value) in results.items():
            frame_id = int(key.split(".")[0])

            for pred in results[key]:
                # FIXME: Filter by car only?
                cls = self.cls2data_id[self.train_id2cls[int(pred[0])]]
                dim = pred[6:9]
                location = pred[9:12]
                ry = [pred[12]]
                score = float(pred[13])
                pred_annos["bbox"].append(location + dim + ry)
                pred_annos["type"].append(cls)
                pred_annos["frame_id"].append(frame_id)
                pred_annos["score"].append(score)

            for gt in self.anns_by_img[frame_id]:
                cls = gt["category_id"]
                dim = gt["dim"]
                location = gt["translation"]
                ry = [gt["rotation_y"]]
                score = 1.0
                gt_annos["bbox"].append(location + dim + ry)
                gt_annos["type"].append(cls)
                gt_annos["frame_id"].append(frame_id)
                gt_annos["score"].append(score)
                gt_annos['diff'].append((2 if (gt['num_lidar'] <= 5 or gt['difficulty'] == 2) else 1))

        return pred_annos, gt_annos

    def get_stats(self, results, save_dir):
        pred_annos, gt_annos = self.get_preds_and_gts(results)

        file_path = os.path.join(save_dir, "eval_results.json")
        with open(file_path, "w") as f:
            json.dump({
                "pred": pred_annos,
                "gt": gt_annos
            }, f)

        command = os.path.join(Path.home(), f"anaconda3/envs/py36_waymo_tf/bin/python -u ultralytics/data/datasets/waymo_eval.py --iou 0.7 --pred {file_path}")
        lines = subprocess.check_output(command, shell= True, text= True, env={})

        print(lines)
        metric3d = float(lines[5].split("|")[2].strip().split(" ")[0])
        return metric3d

    def decode_preds_eval(self, preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=True, threshold=0.001):
        return self.decode_preds(preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=undo_augment, threshold=threshold)

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
                x = bbox[0] * batch["ori_shape"][i][1]  # Always in ori frame because calib is defined there
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
                locations[1] += dimensions[0] / 2

                hd_bin, hd_res = batch["heading_bin"][mask][j].item(), batch["heading_res"][mask][j].item()
                alpha = class2angle(hd_bin, hd_res, to_label_format=True)
                ry = calibs[i].alpha2ry(alpha, x)

                score = 1

                targets.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])

            results[batch["im_file"][i]] = targets
        return results

    def decode_preds(self, preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=True,
                     threshold=0.001):
        preds = preds.detach().cpu()
        bbox, pred_center3d, pred_s3d, pred_hd, pred_dep, pred_dep_un, scores, labels = preds.split(
            (4, 2, 3, 24, 1, 1, 1, 1), dim=-1)

        pred_bin = pred_hd[..., :12]
        pred_res = pred_hd[..., 12:]
        bins = pred_bin.argmax(dim=-1)
        idx = torch.nn.functional.one_hot(pred_bin.max(dim=-1, keepdim=True)[1], num_classes=12).squeeze(-2)
        res = pred_res[idx.bool()].view(bins.shape)
        alphas = bin2angle(bins, res)

        scores = scores.sigmoid()

        results = {}
        for i, img in enumerate(preds):
            targets = []
            for j, pred in enumerate(img):
                cls_id = labels[i, j].item()

                bbox_ = (bbox[i, j].numpy() / np.array([ratio_pad[i][0, 0], ratio_pad[i][0, 1], ratio_pad[i][0, 0],
                                                        ratio_pad[i]
                                                            [0, 1]])).tolist()  # TODO fixme? Shouldn't this be inverse transform?
                x = (bbox_[0] + bbox_[2]) / 2

                dimensions = pred_s3d[i, j].numpy()
                dimensions += self.cls_mean_size[int(cls_id)]

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
                locations[1] += dimensions[0] / 2

                alpha = alphas[i, j].item()
                ry = calibs[i].alpha2ry(alpha, x)

                score = scores[i, j].item() * sigma
                if score < threshold:
                    continue

                targets.append([cls_id, alpha] + bbox_ + dimensions.tolist() + locations.tolist() + [ry, score])

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
            if k in ["img", "coord_range", "ratio_pad", "calib"]:
                value = torch.stack(value, 0)
            if k in ["bboxes", "cls", "depth", "center_3d", "center_2d", "size_2d", "heading_bin",
                     "heading_res", "size_3d"]:
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
