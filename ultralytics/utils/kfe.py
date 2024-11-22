
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ultralytics.utils.keypoint_utils import get_3d_keypoints, rect_to_img

class KFE(nn.Module):
    def __init__(self, in_channels):
        super(KFE, self).__init__()
        self.out_channels = 256
        self.projections = nn.ModuleList(nn.Conv2d(in_ch, 256, 1, 1, (0, 0)) for in_ch in in_channels)

    def forward(self, backbone_features, predictions, calibs, mean_sizes, stride_tensor, strides):
        kps_3d = self.extract_3d_kps(predictions, calibs, mean_sizes)
        kps_2d = rect_to_img(kps_3d, calibs)
        kps_2d = torch.cat((kps_2d,
                            torch.cat((predictions[:, 7].unsqueeze(1).transpose(1, 2).unsqueeze(-1),
                            predictions[:, 8].unsqueeze(1).transpose(1, 2).unsqueeze(-1)), dim=3)), dim=2)
        stride_tensor = stride_tensor.repeat(calibs.shape[0], 1).unsqueeze(-1)
        kps_2d = kps_2d / stride_tensor.unsqueeze(-1)
        features = []
        scale_offset = 0
        for i, stride_ in enumerate(strides):
            bs, ch, h, w = backbone_features[i].shape
            num_kp = kps_2d.shape[2]
            num_anchors_in_scale =  w * h
            proj_features = self.projections[i](backbone_features[i])
            features.append(
                self.bilinear_interpolation(proj_features, kps_2d[:, scale_offset:scale_offset+num_anchors_in_scale])
                    .reshape(bs, self.out_channels, h, w, num_kp))
            scale_offset += num_anchors_in_scale
        return features

    def bilinear_interpolation(self, feature_maps, kps_2d):
        kps_2d[..., 0] = torch.clamp((kps_2d[..., 0] / (feature_maps.shape[3] / 2)) - 1, -1, 1)
        kps_2d[..., 1] = torch.clamp((kps_2d[..., 1] / (feature_maps.shape[2] / 2)) - 1, -1, 1)
        features = nn.functional.grid_sample(feature_maps.float(), kps_2d.float(), mode='nearest', padding_mode="border")
        return features

    def vis_keypoints(self, kps_2d):
        plt.clf()
        image = np.zeros((384, 1280), dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(image)
        ax.scatter(kps_2d[:, 0].detach().cpu(), kps_2d[:, 1].detach().cpu())
        plt.show()

    def extract_3d_kps(self, predictions, calibs, mean_sizes):
        predictions = predictions.transpose(1, 2)
        (cls, bbox, center3d, pred_s3d, pred_hd, pred_dep, pred_dep_un) = predictions.split((3, 4, 2, 3, 24, 1, 1), dim=-1)
        pd_size3d = self.decode_3d_size(pred_s3d, cls, mean_sizes)
        pd_heading_bin, pd_heading_res = pred_hd.split((12, 12), dim=-1)
        return get_3d_keypoints(center3d, pred_dep, pd_size3d, pd_heading_bin, pd_heading_res, calibs)

    def decode_3d_size(self, pd_s3d, pd_scores, mean_sizes):
        pred_class_ind = nn.functional.one_hot(pd_scores.argmax(dim=-1), num_classes=3)
        mean_sizes = mean_sizes.unsqueeze(0).unsqueeze(0).repeat(pd_s3d.shape[0], pd_s3d.shape[1], 1, 1).to(pred_class_ind.device)
        s3d = mean_sizes[pred_class_ind.bool()].reshape(pd_s3d.shape) + pd_s3d
        return s3d
