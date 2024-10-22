# Ultralytics YOLO 🚀, AGPL-3.0 license
import os

import cv2
import torch
import pathlib
from ultralytics.data.datasets.decode_helper import  *
from ultralytics.data.datasets.kitti_eval import eval_from_scrach

import torch.utils.data as data
from PIL import Image

from ultralytics.data.utils import angle2class
from ultralytics.data.datasets.kitti_utils import get_objects_from_label, Calibration, get_affine_transform, affine_transform

from ultralytics.utils.ops import  xyxy2xywh, xywh2xyxy


class KITTIDataset(data.Dataset):
    def __init__(self, image_file_path, mode, args):
        np.random.seed(args.seed)
        # basic configuration
        self.max_objs = 50
        self.class_name = ['Car', 'Pedestrian', 'Cyclist']
        self.cls2train_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
        self.resolution = np.array([1280, 384])  # W * H
        self.use_3d_center = True  # cfg['use_3d_center']
        self.load_depth_maps = args.load_depth_maps
        self.use_camera_dis = args.cam_dis
        self.writelist = ['Car' ,'Pedestrian' ,'Cyclist']

        '''    
        ['Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
         'Pedestrian': np.array([0.84422524,0.66068622,1.76255119]),
         'Cyclist': np.array([1.76282397,0.59706367,1.73698127])] 
        '''
        ##h,l,w
        self.cls_mean_size = np.array([
            [1.52563191462, 1.62856739989, 3.88311640418],
            [1.76255119, 0.66068622, 0.84422524],
            [1.73698127, 0.59706367, 1.76282397]])

        # data split loading
        assert mode in ['train', 'val', 'trainval', 'test']
        self.split = mode
        self.mode = mode
        root_dir = pathlib.Path(image_file_path).parent.parent
        split_dir = image_file_path
        self.idx_list = [x.strip() for x in open(split_dir).readlines()]

        # path configuration
        self.data_dir = os.path.join(root_dir, 'testing' if self.mode == 'test' else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image_2')
        self.depth_dir = os.path.join(root_dir, 'deepseg', "training", "image_2")
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, 'label_2')

        self.im_files = self.get_im_files()
        self.labels = self.get_labels()

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

        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_segmentation(self, idx):
        segmentation_file = os.path.join(self.depth_dir, '%06d_seg.png' % idx)
        assert os.path.exists(segmentation_file)
        return Image.fromarray(cv2.imread(segmentation_file, -1))

    def get_depth_map(self, idx):
        depth_file = os.path.join(self.depth_dir, '%06d_depth.exr' % idx)
        assert os.path.exists(depth_file)
        file = cv2.imread(depth_file, -1)
        return Image.fromarray(np.where(file <= 0, self.max_depth_threshold + 1, file))


    def get_labels(self):
        labels = [self.get_label(int(idx)) for idx in self.idx_list]
        labels = [item for sublist in labels for item in sublist]
        labels = [item for item in labels if item.cls_type in self.writelist]
        return labels

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def get_im_files(self):
        return [os.path.join(self.image_dir, '%06d.png' % int(idx)) for idx in self.idx_list]

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        ori_img = self.get_image(index)
        img = ori_img
        img_size = np.array(ori_img.size)
        if self.split != 'test':
            dst_W, dst_H = img_size

            if self.load_depth_maps:
                depth_maps = []
                seg_mask = self.get_segmentation(index)

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
                if self.load_depth_maps:
                    seg_mask = seg_mask.transpose(Image.FLIP_LEFT_RIGHT)

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
                random_index = np.random.randint(len(self.idx_list))
                random_index = int(self.idx_list[random_index])
                calib_temp = self.get_calib(random_index)

                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    if self.load_depth_maps:
                        seg_mask_tmp = self.get_segmentation(random_index)
                    img_size_temp = np.array(img.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs:
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                                if self.load_depth_maps:
                                    seg_mask_tmp = seg_mask_tmp.transpose(Image.FLIP_LEFT_RIGHT)
                            img = Image.blend(img, img_temp, alpha=0.5)
                            break

        # add affine transformation for 2d images.
        trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
        img = img.transform(tuple(self.resolution.tolist()),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        if self.load_depth_maps:
            seg_mask = np.array(seg_mask.transform(tuple(self.resolution.tolist()),
                                    method=Image.AFFINE,
                                    data=tuple(trans_inv.reshape(-1).tolist()),
                                    resample=Image.NEAREST, fillcolor=51))
            if random_mix_flag == True:
                seg_mask_tmp = np.array(seg_mask_tmp.transform(tuple(self.resolution.tolist()),
                                            method=Image.AFFINE,
                                            data=tuple(trans_inv.reshape(-1).tolist()),
                                            resample=Image.NEAREST, fillcolor=51))

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

            count = 0
            for i in range(object_num):
                # filter objects by writelist
                if objects[i].cls_type not in self.writelist:
                    continue

                # filter inappropriate samples by difficulty
                if objects[i].level_str == 'UnKnown' or (objects[i].pos[-1] * scale < self.min_depth_thres):
                    continue

                if objects[i].trucation > 0.5 or objects[i].occlusion > 2:
                    continue

                # process 2d bbox & get 2d center
                bbox_2d = objects[i].box2d.copy()
                # add affine transformation for 2d boxes.
                bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

                bbox_2d_ = np.copy(bbox_2d)
                bbox_2d_[:2] = bbox_2d[:2]
                bbox_2d_[2:] = bbox_2d[2:]
                bbox_2d_ = xyxy2xywh(bbox_2d_)
                gt_size_2d_ = bbox_2d_[2:]
                center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                     dtype=np.float32)  # W * H

                # process 3d bbox & get 3d center
                center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                r_center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                center_3d, _ = calib.rect_to_img(r_center_3d)  # project 3D center to image plane
                center_3d = center_3d[0]  # shape adjustment
                center_3d = affine_transform(center_3d.reshape(-1), trans)
                gt_center_3d_ = center_3d.copy()

                # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                center_heatmap = center_3d.astype(np.int32)
                if center_heatmap[0] < 0 or center_heatmap[0] >= self.resolution[0]: continue
                if center_heatmap[1] < 0 or center_heatmap[1] >= self.resolution[1]: continue

                # encoding depth
                depth = objects[i].pos[-1]
                depth *= scale
                if depth > self.max_depth_threshold:
                    continue

                cls_id = self.cls2train_id[objects[i].cls_type]
                gt_cls.append([cls_id])
                gt_boxes_2d.append(bbox_2d_)
                gt_center_3d.append(gt_center_3d_)
                gt_center_2d.append(center_2d)
                gt_size_2d.append(gt_size_2d_)

                if self.load_depth_maps:
                    depth_maps.append(np.where(seg_mask == objects[i].line_index, depth, 1000))

                # encoding heading angle
                # heading_angle = objects[i].alpha
                heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                if heading_angle < -np.pi: heading_angle += 2 * np.pi

                heading_bin, heading_res = angle2class(heading_angle)
                gt_heading_bin.append(heading_bin)
                gt_heading_res.append(heading_res)

                s3d = (np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                       - self.cls_mean_size[self.cls2train_id[objects[i].cls_type]])
                gt_size_3d.append(s3d)

                if self.use_camera_dis:
                    r_center_3d *= scale
                    dep = np.linalg.norm(r_center_3d)
                    gt_depth.append(dep)
                else:
                    gt_depth.append(depth)

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
                    if objects[i].cls_type not in self.writelist:
                        continue

                    if objects[i].level_str == 'UnKnown' or (objects[i].pos[-1] * scale < self.min_depth_thres):
                        continue

                    if objects[i].trucation > 0.5 or objects[i].occlusion > 2:
                        continue

                    # process 2d bbox & get 2d center
                    bbox_2d = objects[i].box2d.copy()
                    # add affine transformation for 2d boxes.
                    bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

                    bbox_2d_ = np.copy(bbox_2d)
                    bbox_2d_[:2] = bbox_2d[:2]
                    bbox_2d_[2:] = bbox_2d[2:]
                    bbox_2d_ = xyxy2xywh(bbox_2d_)
                    gt_size_2d_ = bbox_2d_[2:]
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                                         dtype=np.float32)  # W * H

                    # process 3d bbox & get 3d center
                    center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                    r_center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(r_center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    center_3d = affine_transform(center_3d.reshape(-1), trans)

                    # generate the center of gaussian heatmap [optional: 3d center or 2d center]
                    center_heatmap = center_3d.astype(np.int32)
                    if center_heatmap[0] < 0 or center_heatmap[0] >= self.resolution[0]: continue
                    if center_heatmap[1] < 0 or center_heatmap[1] >= self.resolution[1]: continue
                    # encoding depth
                    depth = objects[i].pos[-1]
                    depth *= scale
                    if depth > self.max_depth_threshold:
                        continue

                    cls_id = self.cls2train_id[objects[i].cls_type]
                    gt_cls.append([cls_id])
                    gt_boxes_2d.append(bbox_2d_)
                    gt_center_3d.append(center_3d)
                    gt_center_2d.append(center_2d)
                    gt_size_2d.append(gt_size_2d_)

                    # encoding heading angle
                    heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin, heading_res = angle2class(heading_angle)
                    gt_heading_bin.append(heading_bin)
                    gt_heading_res.append(heading_res)

                    s3d = (np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                           - self.cls_mean_size[self.cls2train_id[objects[i].cls_type]])
                    gt_size_3d.append(s3d)

                    if self.load_depth_maps:
                        depth_maps.append(np.where(seg_mask_tmp == objects[i].line_index, depth, 1000))

                    if self.use_camera_dis:
                        r_center_3d *= scale
                        dep = np.linalg.norm(r_center_3d)
                        gt_depth.append(dep)
                    else:
                        gt_depth.append(depth)

        inputs = torch.tensor(img)
        info = {'img_id': index,
                'img_size': img_size,
                'trans_inv': trans_inv}

        if len(gt_boxes_2d) > 0:
            # We need xywh in [0, 1]
            bboxes = torch.clip(torch.tensor(np.array(gt_boxes_2d) / self.resolution[[0, 1, 0, 1]]), 0, 1)
        else:
            bboxes = torch.empty(0)
        ratio_pad = np.array([self.resolution /img_size, np.array([0, 0])])
        calib = torch.tensor(np.array([calib.cu * ratio_pad[0, 0], calib.cv * ratio_pad[0, 1],
                                       calib.fu * ratio_pad[0, 0], calib.fv * ratio_pad[0, 1],
                                       calib.tx * ratio_pad[0, 0], calib.ty * ratio_pad[0, 1]]))

        if self.load_depth_maps:
            if len(depth_maps) == 0:
                depth_map = torch.tensor(np.zeros_like(seg_mask))
            else:
                depth_map = depth_maps[0]
                for map_ in depth_maps:
                    depth_map = np.minimum(depth_map, map_)
                depth_map = torch.tensor(depth_map)
                depth_map = torch.where(depth_map > self.max_depth_threshold, 0, depth_map)
        else:
            depth_map = torch.empty(1)

        return {
            "img": inputs,
            "ori_img": ori_img,
            "calib": calib,
            "info": info,
            "cls": torch.tensor(np.array(gt_cls)),
            "bboxes": bboxes,
            "batch_idx": torch.zeros(len(gt_boxes_2d)), # Used during collate_fn
            "im_file": '%06d.txt' % info["img_id"],
            "ori_shape": info["img_size"][::-1], # this one is (height, width)
            "ratio_pad": torch.tensor(ratio_pad),
            "center_2d": torch.tensor(np.array(gt_center_2d)),
            "center_3d": torch.tensor(np.array(gt_center_3d)),
            "size_2d": torch.tensor(np.array(gt_size_2d)),
            "size_3d": torch.tensor(np.array(gt_size_3d)),
            "depth": torch.tensor(np.array(gt_depth)),
            "depth_map": depth_map,
            "mean_sizes": torch.tensor(self.cls_mean_size),
            "heading_bin": torch.tensor(np.array(gt_heading_bin)),
            "heading_res": torch.tensor(np.array(gt_heading_res)),
            "mixed": torch.tensor(np.array(random_mix_flag, dtype=np.uint8))
        }

    def get_stats(self, results, save_dir):
        self.save_results(results, output_dir=save_dir)
        result = eval_from_scrach(
            self.label_dir,
            os.path.join(save_dir, 'preds'),
            ap_mode=40)
        return result["3d@0.70"][1]

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'preds')
        os.makedirs(output_dir, exist_ok=True)
        for img_file in results.keys():
            out_path = os.path.join(output_dir, img_file)
            f = open(out_path, 'w')
            for i in range(len(results[img_file])):
                class_name = self.class_name[int(results[img_file][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_file][i])):
                    f.write(' {:.2f}'.format(results[img_file][i][j]))
                f.write('\n')
            f.close()

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

    def decode_preds_eval(self, preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=True,
                          threshold=0.001):
        return self.decode_preds(preds, calibs, im_files, ratio_pad, inv_trans, undo_augment=undo_augment, threshold=threshold)

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

                bbox_ = (bbox[i, j].numpy() / ratio_pad[i][0][[0, 1, 0, 1]]).tolist()
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
                    x3d = pred_center3d[i, j, 0].numpy() * 1242 / 1280.0
                    y3d = pred_center3d[i, j, 1].numpy() * 375 / 384.0
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
            if k in ["img", "coord_range", "ratio_pad", "calib", "mixed", "depth_map"]:
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
