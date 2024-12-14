# Ultralytics YOLO 🚀, AGPL-3.0 license
import os

import torch.utils.data as data
from PIL import Image
import subprocess
from pathlib import Path

from scipy.spatial.transform import Rotation

from ultralytics.utils.keypoint_utils import *
from ultralytics.utils.ops import *

from collections import defaultdict
import json

from ultralytics.data.datasets.kitti_utils import get_objects_from_dict_rope, Calibration, get_affine_transform, affine_transform


class Rope3Dataset(data.Dataset):
    def __init__(self, filepath, mode, args):
        self.args = args
        self.path = "/".join(filepath.split("/")[:-1])
        self.split = mode
        self.mode = mode
        self.class_name = ['Car', 'Pedestrian', 'Cyclist']
        self.writelist = ['Car', 'Pedestrian', 'Cyclist']
        self.resolution = np.array([960, 540])  # W * H
        self.max_objs = 80
        self.use_camera_dis = False
        self.load_depth_maps = False

        print("Loading Rope3D Dataset...")
        self.raw_split = json.load(open(filepath, 'r'))
        if args.overfit:
            self.raw_split["images"] = [image for image in self.raw_split["images"] if image["id"] < 50]
            self.raw_split["annotations"] = [anns for anns in self.raw_split["annotations"] if anns["image_id"] < 50]

        self.imgs = {img['id']: img for img in sorted(self.raw_split['images'], key=lambda img: img['id'])}
        self.idx_to_img_id = {idx: img_id for idx, img_id in enumerate(self.imgs)}
        self.img_file2img_id = {img["file_path"].split(os.path.sep)[-1]: idx for idx, img in self.imgs.items()}

        self.cls2train_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        self.train_id2cls = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}

        self.data_cls2data_id = {value["name"].title(): value["id"] for value in self.raw_split["categories"]}
        self.data_id2data_cls = {cls_id: cls_name for cls_name, cls_id in self.data_cls2data_id.items()}

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
        self.rotation = args.rotation
        self.virtual_focal_length = args.virtual_focal_length

        assert(self.pred_rot_mat)

        self.left_multiply_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.right_multiply_matrix = np.array([[-1, 0, 0], [0, 1, 0.], [0, 0, -1]])

        print("Finished loading!")

    def get_image(self, idx):
        img_file = os.path.join(self.path, self.imgs[idx]["file_path"].replace("images/", ""))
        if not os.path.exists(img_file):
            print(f"Missing segment: {img_file}")
        return Image.open(img_file).convert("RGB")

    def get_label(self, idx):
        return get_objects_from_dict_rope(self.anns_by_img[idx])

    def get_labels(self):
        labels = self.anns_by_img
        labels = [item for sublist in labels.values() for item in sublist]
        labels = [item for item in labels if item["category"] in self.writelist]
        return labels

    def get_calib(self, idx):
        calib = np.array(self.imgs[idx]["K"])
        calib = np.hstack((calib, np.zeros((3, 1))))
        return Calibration({"P2": calib, 'R0': 0, "Tr_velo2cam": np.ones_like(calib)})
    
    def get_c2g(self, idx):
        return np.array(self.imgs[idx]["c2g_trans"])

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
        random_rot_flag = False
        calib = self.get_calib(index)
        rot_angle = 0
        scale = 1
        shift = np.array([0, 0])
        vdepth_factor0 = self.virtual_focal_length / calib.fv

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

            if np.random.random() < self.rotation:
                random_rot_flag = True
                rot_angle = int(np.random.uniform(-180, 180))

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = np.random.randint(self.__len__())
                random_index = int(self.idx_to_img_id[random_index])
                calib_temp = self.get_calib(random_index)
                vdepth_factor1 = self.virtual_focal_length / calib_temp.fv
                
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
        trans, trans_inv = get_affine_transform(center, crop_size, rot_angle, self.resolution, inv=1)
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
        gt_vdep_factors = []

        if self.split != 'test':
            objects = self.get_label(index)
            # data augmentation for labels
            if random_flip_flag:
                calib.horizontal_flip(img_size)
                for object in objects:
                    [x1, _, x2, _] = object.box2d # xyxy
                    object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    object.pos[0] *= -1 # object position is expressed in camera coordinates
                    object.rot_mat = self.left_multiply_matrix @ object.rot_mat @ self.right_multiply_matrix
            if random_rot_flag:
                pass


            object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

            for i in range(object_num):
                valid, _box, _cls, _center2d, _center3d, _size2d, _size3d, _depth, _head_bin, _head_res, _rot_mat \
                    = self.load_object(objects[i], scale, trans, calib, shift, ratio_pad, rot_angle)
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
                    gt_vdep_factors.append(vdepth_factor0)

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
                        = self.load_object(objects[i], scale, trans, calib, shift, ratio_pad, rot_angle)
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
                        gt_vdep_factors.append(vdepth_factor1)

        inputs = torch.tensor(img)
        info = {'img_id': index,
                'img_file': self.imgs[index]["file_path"].split(os.path.sep)[-1],
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
            "im_file": info["img_file"],
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
            "shift": torch.tensor(shift),
            "vdepth_factors": torch.tensor(gt_vdep_factors)
        }
        return data

    def load_object(self, object_, scale, trans, calib, shift, ratio_pad, rot_angle):
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
        center_3d_pre = center_3d[0]  # shape adjustment
        center_3d = affine_transform(center_3d_pre.reshape(-1), trans)
        _center3d = center_3d.copy()

        # process 2d bbox & get 2d center
        bbox_2d = object_.box2d
        # add affine transformation for 2d boxes.


        # if the image was rotated, reorient the bounding box
        corner_points = bbox_2d[[0, 1, 2, 3, 0, 3, 2, 1]].copy().reshape(4, 2)
        bbox_2d_rot = np.zeros(8)
        bbox_2d_rot[:2] = affine_transform(corner_points[0], trans)
        bbox_2d_rot[2:4] = affine_transform(corner_points[1], trans)
        bbox_2d_rot[4:6] = affine_transform(corner_points[2], trans)
        bbox_2d_rot[6:8] = affine_transform(corner_points[3], trans)
        corner_xs = bbox_2d_rot[0::2]
        corner_ys = bbox_2d_rot[1::2]
        bbox_2d = np.array([corner_xs.min(), corner_ys.min(), corner_xs.max(), corner_ys.max()])

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

        egoc_rot_matrix = object_.rot_mat
        left_multiply_matrix = Rotation.from_euler('XYZ', [0, 0, -rot_angle / 180 * np.pi]).as_matrix()
        egoc_rot_matrix = left_multiply_matrix @ egoc_rot_matrix

        in_pos = np.array([_center3d[0] + shift[0]*trans[0, 0], _center3d[1] + shift[1]*trans[1, 1]])
        _rot_mat = egoc_to_alloc_rot_matrix_torch(amodal_center=torch.from_numpy(in_pos).unsqueeze(0).float(),
                                                  egoc_rot_matrix=torch.from_numpy(egoc_rot_matrix).unsqueeze(0).float(),
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

    @staticmethod
    def get_3d_box(location, rotation, size3d):
        boxes_object_frame = get_omni_eval_corners(size3d)
        return transform_to_camera(boxes_object_frame, location, rotation)

    def save_preds(self, results, output_dir):
        output_dir = os.path.join(output_dir, 'preds')
        os.makedirs(output_dir, exist_ok=True)
        for img_file in results.keys():
            out_path = os.path.join(output_dir, img_file.split(".")[0] + ".txt")
            f = open(out_path, 'w')
            for i in range(len(results[img_file])):
                class_name = self.class_name[int(results[img_file][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_file][i])):
                    f.write(' {:.2f}'.format(results[img_file][i][j]))
                f.write('\n')
            f.close()
        return output_dir

    def get_stats(self, results, save_dir):
        output_dir = self.save_preds(results, save_dir)

        command = (f"source /home/stud/mijo/dev/rope3d_eval/evaluate.sh {output_dir}")
        out = subprocess.check_output(command, shell= True, text= True, executable="/bin/bash")
        lines = out.split("\n")
        
        metric3d = float(lines[11].split(" ")[3].strip()) # 0.7 moderate
        #metric3d = float(lines[12].split(" ")[3].strip()) # 0.7 hard
        #metric3d = float(lines[14].split(" ")[3].strip()) # 0.5 moderate
        #metric3d = float(lines[15].split(" ")[3].strip()) # 0.5 hard
        return metric3d
    
    def egoc_rot_matrix2rot_y(self, c2g_trans: np.ndarray, egoc_rot_matrix: np.ndarray) -> torch.Tensor:
        """
        Rope3D provides a single angle per object only and the ground equation.
        From the ground equation we can compute c2g_trans.
        We can then use c2g_trans and the egocentric rotation inferred by the network to compute rot_y again.
        See: https://github.com/liyingying0113/rope3d-dataset-tools/blob/main/show_tools/show_2d3d_box.py
        Args:
        c2g_trans (np.ndarray): B x 3 x 3. Rotation matrix to map between ground and camera
        egoc_rot_matrix (np.ndarray): B x 3 x 3. Egocentric rotation matrix of the object
        Returns:
        np.ndarray: B. Rotation of the object around the y-axis of the camera
        """
        ground_r_recovered = c2g_trans @ egoc_rot_matrix
        _, _, yaw_world_res_recovered = Rotation.from_matrix(ground_r_recovered).as_euler('XYZ')
        R = c2g_trans[:3, :3]
        theta_recovered_v1 = np.arctan((R[1, 0] / np.tan(yaw_world_res_recovered) - R[0, 0]) /
        (R[1, 2] / np.tan(yaw_world_res_recovered) - R[0, 2]))
        theta_recovered_v2 = theta_recovered_v1 + np.pi
        beta_recovered_v1 = np.arcsin(R[2, 2] * np.sin(theta_recovered_v1) - R[2, 0] * np.cos(theta_recovered_v1))
        beta_recovered_v2 = np.arcsin(R[2, 2] * np.sin(theta_recovered_v2) - R[2, 0] * np.cos(theta_recovered_v2))
        target_theta0_1_recovered_v1 = np.cos(beta_recovered_v1) * np.sin(yaw_world_res_recovered)
        target_theta0_1_recovered_v2 = np.cos(beta_recovered_v2) * np.sin(yaw_world_res_recovered)
        theta0_1_recovered_v1 = R[1, 0] * np.cos(theta_recovered_v1) - R[1, 2] * np.sin(theta_recovered_v1)
        theta0_1_recovered_v2 = R[1, 0] * np.cos(theta_recovered_v2) - R[1, 2] * np.sin(theta_recovered_v2)
        assert np.isclose(target_theta0_1_recovered_v1, target_theta0_1_recovered_v2)
        if np.isclose(theta0_1_recovered_v1, target_theta0_1_recovered_v1):
            theta_recovered = theta_recovered_v1
        elif np.isclose(theta0_1_recovered_v2, target_theta0_1_recovered_v1):
            theta_recovered = theta_recovered_v2

        return theta_recovered

    def decode_preds_eval(self, preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=True,
                          threshold=0.001, ground_center=True):
        return self.decode_preds(preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=undo_augment,
                                 threshold=threshold, ground_center=ground_center, roty=True)

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
                     threshold=0.001, ground_center=True, roty=False):
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

                depth = pred_dep[i, j].numpy() / (self.virtual_focal_length / calibs[i].fv)
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

                if roty:
                    c2g_trans = self.get_c2g(self.img_file2img_id[im_files[i].split(os.path.sep)[-1]])
                    roty = self.egoc_rot_matrix2rot_y(c2g_trans, egoc_rot_mat)
                    targets.append([cls_id, roty] + bbox + dimensions.tolist() + locations.tolist() + [score])
                else:
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
                     "heading_res", "size_3d", "rot_mat", "vdepth_factors"]:
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
