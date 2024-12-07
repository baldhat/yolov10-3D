
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from ultralytics.nn.modules import Conv
from ultralytics.utils.keypoint_utils import get_3d_keypoints, rect_to_img


class KFE(nn.Module):
    def __init__(self, in_channels):
        super(KFE, self).__init__()
        self.out_channels = 256
        self.num_keypoints = 9
        self.projections = nn.ModuleList(nn.Conv2d(in_ch, 256, 1, 1, (0, 0)) for in_ch in in_channels)
        self.keypoints_head = nn.Sequential(
            Conv(256, 128),
            Conv(128, 64),
            Conv(64, 24, act=torch.nn.Tanh())
        )

    def pos2posemb3d(self, keypoint_positions, num_pos_feats=128, temperature=10000):
        pos_embeddings = []
        for keypoint_pos in keypoint_positions:
            scale = torch.pi
            keypoint_pos = keypoint_pos * scale
            dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=keypoint_pos.device)
            dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
            pos_x = keypoint_pos[:, 0::3, :, :, None] / dim_t
            pos_y = keypoint_pos[:, 1::3, :, :, None] / dim_t
            pos_z = keypoint_pos[:, 2::3, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
            posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1).transpose(1, -1)
            pos_embeddings.append(posemb)
        return pos_embeddings

    def forward(self, backbone_features, predictions, calibs, mean_sizes, stride_tensor, strides):
        proj_features = []
        keypoints_rel_pos = []
        for i, _ in enumerate(strides):
            proj_features.append(self.projections[i](backbone_features[i]))
            pos = self.keypoints_head(proj_features[i])
            pos = torch.cat((pos, torch.zeros(pos.shape[0], 3, pos.shape[2], pos.shape[3], device=pos.device)),
                            dim=1)
            keypoints_rel_pos.append(pos)

        bs = proj_features[0].shape[0]
        kps_input = torch.cat([krp.reshape(bs, -1, self.num_keypoints * 3).reshape(bs, -1, self.num_keypoints, 3) for krp in keypoints_rel_pos], dim=1)[:, :, :8]
        kps_3d = self.extract_3d_kps(predictions, calibs, mean_sizes,
                                     kps_input)
        kps_2d = rect_to_img(kps_3d, calibs)
        pred_center_x_idx, pred_center_y_idx = 7, 8
        center3d = torch.cat((predictions[:, pred_center_x_idx].unsqueeze(1).transpose(1, 2).unsqueeze(-1),
                   predictions[:, pred_center_y_idx].unsqueeze(1).transpose(1, 2).unsqueeze(-1)), dim=3)
        kps_2d = torch.cat((kps_2d, center3d), dim=2)
        stride_tensor = stride_tensor.repeat(calibs.shape[0], 1).unsqueeze(-1)
        kps_2d = kps_2d / stride_tensor.unsqueeze(-1)
        features = []
        scale_offset = 0
        for i, stride_ in enumerate(strides):
            bs, ch, h, w = backbone_features[i].shape
            num_kp = kps_2d.shape[2]
            num_anchors_in_scale =  w * h
            features.append(
                self.bilinear_interpolation(proj_features[i], kps_2d[:, scale_offset:scale_offset+num_anchors_in_scale])
                    .reshape(bs, self.out_channels, h, w, num_kp))
            scale_offset += num_anchors_in_scale
        positional_embedding = self.pos2posemb3d(keypoints_rel_pos)
        return features, positional_embedding

    def bilinear_interpolation(self, feature_maps, kps_2d):
        kps_2d[..., 0] = torch.clamp((kps_2d[..., 0] / (feature_maps.shape[3] / 2)) - 1, -1, 1)
        kps_2d[..., 1] = torch.clamp((kps_2d[..., 1] / (feature_maps.shape[2] / 2)) - 1, -1, 1)
        return nn.functional.grid_sample(feature_maps.float(), kps_2d.float(), mode='bilinear', padding_mode="border")

    def vis_keypoints(self, kps_2d):
        plt.clf()
        image = np.zeros((384, 1280), dtype=float)
        fig, ax = plt.subplots(1, 1, figsize=(50, 20))
        ax.imshow(image)
        if len(kps_2d.shape) > 2:
            for anchor in kps_2d:
                ax.scatter(anchor[:8, 0].detach().cpu(), anchor[:8, 1].detach().cpu(), color=(0, 0, 1))
                ax.scatter(anchor[8, 0].detach().cpu(), anchor[8, 1].detach().cpu(), color=(1, 0, 0))
        else:
            ax.scatter(kps_2d[:8, 0].detach().cpu(), kps_2d[:8, 1].detach().cpu(), color=(0,0,1))
            ax.scatter(kps_2d[8, 0].detach().cpu(), kps_2d[8, 1].detach().cpu(), color=(1,0,0))
        fig.tight_layout()
        plt.show()

    def extract_3d_kps(self, predictions, calibs, mean_sizes, kps_rel_pos):
        predictions = predictions.transpose(1, 2)
        (cls, bbox, center3d, pred_s3d, pred_hd, pred_dep, pred_dep_un) = predictions.split((3, 4, 2, 3, 24, 1, 1), dim=-1)
        pd_size3d = self.decode_3d_size(pred_s3d, cls, mean_sizes)
        pd_heading_bin, pd_heading_res = pred_hd.split((12, 12), dim=-1)
        return get_3d_keypoints(center3d, pred_dep, pd_size3d, pd_heading_bin, pd_heading_res, calibs, kps_rel_pos)

    def decode_3d_size(self, pd_s3d, pd_scores, mean_sizes):
        pred_class_ind = nn.functional.one_hot(pd_scores.argmax(dim=-1), num_classes=3)
        mean_sizes = mean_sizes.unsqueeze(0).unsqueeze(0).repeat(pd_s3d.shape[0], pd_s3d.shape[1], 1, 1).to(pred_class_ind.device)
        s3d = mean_sizes[pred_class_ind.bool()].reshape(pd_s3d.shape) + pd_s3d
        return s3d
