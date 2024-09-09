import numpy as np
import torch
import torch.nn as nn
from ultralytics.utils import ops
from ultralytics.data.kitti_utils import affine_transform


def class2angle(cls, residual, to_label_format=False, num_heading_bin = 12):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle

def bin2angle(cls, residual, num_heading_bin = 12):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    angle[angle > np.pi] = angle[angle > np.pi] - 2 * np.pi
    return angle

def decode_batch(batch, calibs, cls_mean_size, use_camera_dis=False, undo_augment=True):
    results = {}
    for i in range(batch["img"].shape[0]):
        targets = []
        mask = batch["batch_idx"] == i
        num_targets = mask.sum()
        for j in range(num_targets):
            cls_id = batch["cls"][mask][j].item()

            bbox = batch["bboxes"][mask][j].cpu().numpy()
            x = bbox[0] * batch["ori_shape"][i][1] # Always in ori frame because calib is defined there
            bbox = (ops.xywh2xyxy(bbox)  * batch["ori_shape"][i][[1, 0, 1, 0]]).tolist()

            dimensions = batch["size_3d"][mask][j].cpu().numpy()
            dimensions += cls_mean_size[int(cls_id)]

            depth = batch["depth"][mask][j].cpu().numpy()

            if undo_augment:
                x3d = batch["center_3d"][mask][j, 0].cpu().numpy()
                y3d = batch["center_3d"][mask][j, 1].cpu().numpy()
                c3d = affine_transform(np.array([x3d, y3d]), np.array(batch["info"][i]["trans_inv"]))
                if use_camera_dis:
                    locations = calibs[i].camera_dis_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                else:
                    locations = calibs[i].img_to_rect(c3d[0], c3d[1], depth).reshape(-1)
            else:
                x3d = batch["center_3d"][mask][j, 0].cpu().numpy() * 1242/1280
                y3d = batch["center_3d"][mask][j, 1].cpu().numpy() * 375/384.0
                if use_camera_dis:
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



def decode_preds(preds, calibs, cls_mean_size, im_files, inv_trans, use_camera_dis, undo_augment=True, threshold=0.001, nms=False):
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

    if nms:
        preds = ops.non_max_suppression(torch.cat((bbox, scores,
                                                 torch.cat((pred_center3d, pred_s3d, alphas.unsqueeze(-1), pred_dep, pred_dep_un, labels), dim=-1)),
                                                dim=-1).permute(0, 2, 1),
                                  xyxy=True, iou_thres=0.8, conf_thres=0.001, agnostic=True, nc=1)

    results = {}
    for i, img in enumerate(preds):
        if nms:
            bbox, scores, _, pred_center3d, pred_s3d, alphas, pred_dep, pred_dep_un, labels = preds[i].split((4, 1, 1, 2, 3, 1, 1, 1, 1),
                                                                                             dim=-1)
            targets = []
            for j, pred in enumerate(img):
                cls_id = labels[j].item()

                bbox_ = (bbox[j].numpy() * np.array([1242 / 1280.0, 375 / 384.0, 1242 / 1280.0,
                                                        375 / 384.0])).tolist()  # TODO fixme? Shouldn't this be inverse transform?
                x = (bbox_[0] + bbox_[2]) / 2

                dimensions = pred_s3d[j].numpy()
                dimensions += cls_mean_size[int(cls_id)]

                depth = pred_dep[j].numpy()
                sigma = torch.exp(-pred_dep_un[j])

                if undo_augment:
                    x3d = pred_center3d[j, 0].numpy()
                    y3d = pred_center3d[j, 1].numpy()
                    c3d = affine_transform(np.array([x3d, y3d]), np.array(inv_trans[i]))
                    if use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                else:
                    x3d = pred_center3d[j, 0].numpy() * 1242 / 1280.0
                    y3d = pred_center3d[j, 1].numpy() * 375 / 384.0
                    if use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(x3d, y3d, depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2

                alpha = alphas[j].item()
                ry = calibs[i].alpha2ry(alpha, x)

                score = scores[j].item() * sigma.item()
                if score < threshold:
                    continue

                targets.append([cls_id, alpha] + bbox_ + dimensions.tolist() + locations.tolist() + [ry, score])

            results[im_files[i]] = targets
        else:
            targets = []
            for j, pred in enumerate(img):
                cls_id = labels[i, j].item()

                bbox_ = (bbox[i, j].numpy() * np.array([1242/1280.0, 375/384.0, 1242/1280.0, 375/384.0])).tolist() #TODO fixme? Shouldn't this be inverse transform?
                x = (bbox_[0] + bbox_[2]) / 2

                dimensions = pred_s3d[i, j].numpy()
                dimensions += cls_mean_size[int(cls_id)]

                depth = pred_dep[i, j].numpy()
                sigma = torch.exp(-pred_dep_un[i, j])

                if undo_augment:
                    x3d = pred_center3d[i, j, 0].numpy()
                    y3d = pred_center3d[i, j, 1].numpy()
                    c3d = affine_transform(np.array([x3d, y3d]), np.array(inv_trans[i]))
                    if use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(c3d[0], c3d[1], depth).reshape(-1)
                else:
                    x3d = pred_center3d[i, j, 0].numpy() * 1242/1280.0
                    y3d = pred_center3d[i, j, 1].numpy() * 375/384.0
                    if use_camera_dis:
                        locations = calibs[i].camera_dis_to_rect(x3d, y3d, depth).reshape(-1)
                    else:
                        locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
                locations[1] += dimensions[0] / 2

                alpha = alphas[i, j].item()
                ry = calibs[i].alpha2ry(alpha, x)

                score = scores[i, j].item() * sigma.item()
                if score < threshold:
                    continue

                targets.append([cls_id, alpha] + bbox_ + dimensions.tolist() + locations.tolist() + [ry, score])

            results[im_files[i]] = targets
    return results

def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)


