import numpy as np
import torch
import torch.nn as nn


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

def decode_batch(batch, calibs, cls_mean_size):
    results = {}
    for i in range(batch["img"].shape[0]):
        targets = []
        mask = batch["batch_idx"] == i
        num_targets = mask.sum()
        for j in range(num_targets):
            cls_id = batch["cls"][mask][j].item()

            bbox = batch["bboxes"][mask][j].cpu().numpy().tolist()

            x = bbox[0] * batch["img"].shape[-1]

            dimensions = batch["size_3d"][mask][j].cpu().numpy()
            dimensions += cls_mean_size[int(cls_id)]

            depth = batch["depth"][mask][j].cpu().numpy()

            x3d = batch["center_3d"][mask][j, 0].cpu().numpy()
            y3d = batch["center_3d"][mask][j, 1].cpu().numpy()
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            hd_bin, hd_res = batch["heading_bin"][mask][j].item(), batch["heading_res"][mask][j].item()
            alpha = class2angle(hd_bin, hd_res, to_label_format=True)
            ry = calibs[i].alpha2ry(alpha, x)  # FIXME: is this correct? is this all we need?

            score = 1

            targets.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])

        results[batch["im_file"][i]] = targets
    return results



def decode_preds(preds, calibs, cls_mean_size, im_files, threshold=0.001):
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
            if scores[i, j].item() < threshold:
                continue
            cls_id = labels[i, j].item()

            bbox_ = bbox[i, j].numpy().tolist()
            x = bbox_[0]

            dimensions = pred_s3d[i, j].numpy()
            dimensions += cls_mean_size[int(cls_id)]

            depth = pred_dep[i, j].numpy()

            x3d = pred_center3d[i, j, 0].numpy() * 1224/1280.0
            y3d = pred_center3d[i, j, 1].numpy() * 1224/1280.0
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            alpha = alphas[i, j].item()
            ry = calibs[i].alpha2ry(alpha, x)  # FIXME: is this correct? is this all we need?

            score = scores[i, j].item() # FIXME: include depth uncertainty

            targets.append([cls_id, alpha] + bbox_ + dimensions.tolist() + locations.tolist() + [ry, score])

        results[im_files[i]] = targets
    return results



def decode_detections(dets, info, calibs, cls_mean_size, threshold, problist=None):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''

    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold: continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            depth = dets[i, j, -2]
            score *= dets[i, j, -1]

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 6:30])
            ry = calibs[i].alpha2ry(alpha, x)

            # dimensions decoding
            dimensions = dets[i, j, 30:33]
            dimensions += cls_mean_size[int(cls_id)]
            if True in (dimensions<0.0): continue

            # positions decoding
            x3d = dets[i, j, 33] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


# def convert_color_map(img, mode=cv.INTER_LINEAR, size=49):
#     # temp = img / np.max(img) * 255
#     # temp = img / 50 * 255
#     temp = cv.resize(img.cpu().numpy(), size, interpolation=mode) * 250
#     temp = temp.astype(np.uint8)
#     im_color = cv.applyColorMap(temp, cv.COLORMAP_JET)
#     return im_color

#two stage style
def extract_dets_from_outputs(outputs, conf_mode='ada', K=50):
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    batch, channel, height, width = heatmap.size() # get shape
    heading = outputs['heading'].view(batch,K,-1)

    vis_depth = outputs['vis_depth'].view(batch,K,7,7)
    ins_depth = vis_depth
    ins_depth_uncer = outputs['vis_depth_uncer'].view(batch,K,7,7)
    merge_prob = (-(0.5 * ins_depth_uncer).exp()).exp()
    merge_depth = (torch.sum((ins_depth*merge_prob).view(batch,K,-1), dim=-1) /
                   torch.sum(merge_prob.view(batch,K,-1), dim=-1))
    merge_depth = merge_depth.unsqueeze(2)


    ins_depth_test = ins_depth.view(batch,K,-1)
    merge_prob_max_ind = torch.argmax(merge_prob.view(batch,K,-1),dim=-1)
    ins_depth_test = ins_depth_test.view(-1,49)
    merge_prob_max_ind = merge_prob_max_ind.view(-1,1)
    ins_depth_max = torch.gather(ins_depth_test,1,index = merge_prob_max_ind).view(batch,K,-1)
    # merge_depth = ins_depth_max

    ins_depth_uncer = outputs['attention_map'].view(batch,K,7,7)
    ins_depth_test = ins_depth.view(batch,K,-1)
    ins_depth_uncer_ind = torch.argmax(ins_depth_uncer.view(batch,K,-1),dim=-1)
    ins_depth_test = ins_depth_test.view(-1,49)
    ins_depth_uncer_ind = ins_depth_uncer_ind.view(-1,1)
    ins_depth_max = torch.gather(ins_depth_test,1,index = ins_depth_uncer_ind).view(batch,K,-1)
    merge_depth = ins_depth_max


    if conf_mode == 'ada':
        merge_conf = (torch.sum(merge_prob.view(batch,K,-1)**2, dim=-1) / \
                      torch.sum(merge_prob.view(batch,K,-1), dim=-1)).unsqueeze(2)
    elif conf_mode == 'max':
        merge_conf = (merge_prob.view(batch, K, -1).max(-1))[0].unsqueeze(2)
    else:
        raise NotImplementedError("%s confidence aggreation is not supported" % conf_mode)

    size_3d = outputs['size_3d'].view(batch,K,-1)
    offset_3d = outputs['offset_3d'].view(batch,K,-1)

    heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, heading, size_3d, xs3d, ys3d, merge_depth, merge_conf], dim=2)

    return detections


def extract_dets_from_targets(targets, K=50):
    heatmap = torch.tensor(targets['heatmap'])

    batch, channel, height, width = heatmap.size() # get shape
    heading_ = nn.functional.one_hot(torch.tensor(targets['heading_bin']), num_classes=12).squeeze()
    heading = torch.zeros_like(heading_)
    heading_res_ = torch.zeros_like(heading_, dtype=torch.float)
    heading_res = torch.zeros_like(heading_res_)
    heading_res_[torch.nonzero(heading_, as_tuple=True)] = torch.tensor(targets["heading_res"]).flatten()

    depth = torch.zeros_like(torch.tensor(targets["depth"]))
    depth_conf = torch.ones_like(depth)

    size_3d_ = torch.tensor(targets['size_3d']).view(batch,K,-1)
    size_3d = torch.zeros_like(size_3d_)
    offset_3d_ = torch.tensor(targets['offset_3d']).view(batch,K,-1)
    offset_3d = torch.zeros_like(offset_3d_)

    #heatmap = torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    size_2d = torch.zeros((batch, 2, height, width))
    offset_2d = torch.zeros((batch, 2, height, width))
    for b in range(batch):
        indices = inds[b]
        t_indices = targets["indices"][b]
        for i, t_ix in enumerate(t_indices):
            if t_ix != 0:
                index = ((indices == t_ix).nonzero(as_tuple=True)[0])
                x, y = int(xs[b, index].item()), int(ys[b, index].item())
                size_2d[b, :, y, x] = torch.tensor(targets['size_2d'])[b, i]
                offset_2d[b, :, y, x] = torch.tensor(targets['offset_2d'])[b, i]
                depth[b, index] = torch.tensor(targets["depth"][b, i])
                heading[b, index] = heading_[b, i]
                heading_res[b, index] = heading_res_[b, i]
                size_3d[b, index] = size_3d_[b, i]
                offset_3d[b, index] = offset_3d_[b, i]

    heading = torch.cat((heading, heading_res), dim=-1)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)

    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]

    xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
    ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, heading, size_3d, xs3d, ys3d, depth, depth_conf], dim=2)

    return detections

def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding)
    keep = (heatmapmax == heatmap).float()
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    if torch.__version__ == '1.6.0':
        topk_ys = (topk_inds // width).int().float()
    else:
        topk_ys = (topk_inds / width).int().float()
    # topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    if torch.__version__ == '1.6.0':
        topk_cls_ids = (topk_ind // K).int()
    else:
        topk_cls_ids = (topk_ind / K).int()
    # topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)


