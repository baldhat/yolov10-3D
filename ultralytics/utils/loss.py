# Ultralytics YOLO 🚀, AGPL-3.0 license
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (RotatedTaskAlignedAssigner, TaskAlignedAssigner, TaskAlignedAssigner3d,
                                   dist2bbox, dist2rbox, make_anchors)
from .metrics import bbox_iou, probiou
from .tal import bbox2dist

import numpy as np
import matplotlib.pyplot as plt
import cv2


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25, reduction="none"):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction=reduction)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

class v10DetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)
    
    def __call__(self, preds, batch):
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], torch.cat((loss_one2many[1], loss_one2one[1]))


class DetectLoss3d:
    def __init__(self, model):
        self.one2many = DDDetectionLoss(model, tal_topk=model.args.tal_topk)
        self.one2one = DDDetectionLoss(model, tal_topk=1)
        self.model = model
        if self.model.args.fgdm_loss:
            self.fgdm_loss_func = ForegroundDepthMapLoss(self.model)

    def __call__(self, preds, batch):
        one2one, o2o_embs = preds["one2one"], preds["o2o_embs"]
        loss_one2one = self.one2one(one2one, batch, embeddings=o2o_embs)

        if preds.get("one2many", None):
            one2many, o2m_embs = preds["one2many"], preds["o2m_embs"]
            loss_one2many = self.one2many(one2many, batch, embeddings=o2m_embs)

            if self.model.args.fgdm_loss:
                gt_depth_maps = batch["depth_map"].to(preds["depth_maps"][0].device)
                depth_logits = preds["depth_maps"][0]
                fgdm_loss = self.fgdm_loss_func(depth_logits, gt_depth_maps) * self.model.args.fgdm_loss_weight
                return loss_one2many[0] + loss_one2one[0] + fgdm_loss, torch.cat((loss_one2many[1], loss_one2one[1], fgdm_loss.unsqueeze(0)))
            else:
                return loss_one2many[0] + loss_one2one[0], torch.cat((loss_one2many[1], loss_one2one[1]))
        else:
            return torch.zeros(1), torch.cat((loss_one2one[1], torch.zeros_like(loss_one2one[1])))


class DDDetectionLoss:
    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device

        self.assigner = TaskAlignedAssigner3d(topk=tal_topk, num_classes=self.nc,
                                              alpha=model.args.tal_alpha, beta=model.args.tal_beta,
                                              gamma=model.args.tal_gamma, use_2d=model.args.tal_2d,
                                              use_3d=model.args.tal_3d, kps_dist_metric=model.args.kps_dist_metric,
                                              constrain_anchors=model.args.constrain_anchors)
        if self.hyp.distillation:
            self.supervisor = SupervisionLoss(model)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 17, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 17, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_2d, stride_tensor):
        # anchor_points:
        # pred_2d: offset_2d (2), size_2d(2)
        offset, size = pred_2d.split((2, 2), dim=-1)
        centers = anchor_points + offset
        xy1 = centers - size / 2
        xy2 = centers + size / 2
        return torch.cat((xy1, xy2), dim=-1) * stride_tensor

    def __call__(self, preds, batch, embeddings):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(7 if self.hyp.distillation else 6, device=self.device)  # box, cls, dep, o3d, s3d, hd
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_scores, pred_o2d, pred_s2d, pred_o3d, pred_s3d, pred_hd, pred_dep, pred_dep_un = (
            torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.nc, 2, 2, 2, 3, 24, 1, 1), 1
        ))

        # num classes
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        # offset 2d (2), size 2d (2) = 4
        pred_2d = torch.cat((pred_o2d.permute(0, 2, 1).contiguous(), pred_s2d.permute(0, 2, 1).contiguous()), -1)
        # offset 3d (2), size 3d (3) = 5
        pred_3d = torch.cat((pred_o3d.permute(0, 2, 1).contiguous(),     # offset 3d (2)
                             pred_s3d.permute(0, 2, 1).contiguous(),            # size 3d (3)
                             pred_hd.permute(0, 2, 1).contiguous(),             # heading bins (12) + heading res (12)
                             pred_dep.permute(0, 2, 1).contiguous(),            # depth (1)
                             pred_dep_un.permute(0, 2, 1).contiguous()), -1)    # depth uncertainty (1)
        # = 38

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        gts = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1),
                                batch["bboxes"], batch["center_2d"], batch["size_2d"],
                                batch["center_3d"], batch["size_3d"], batch["depth"].view(-1, 1),
                                batch["heading_bin"].view(-1, 1), batch["heading_res"].view(-1, 1)), 1)
        gts = self.preprocess(gts.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        calibs = batch["calib"]
        mean_sizes = batch["mean_sizes"]
        gt_labels, gt_bboxes, gt_center_2d, gt_size_2d, gt_center_3d, gt_size_3d, gt_depth, gt_heading_bin, gt_heading_res = gts.split(
            (1, 4, 2, 2, 2, 3, 1, 1, 1), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_2d, stride_tensor)

        targets, fg_mask, target_gt_idx, pred_kps, gt_kps = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach().type(gt_bboxes.dtype),
            pred_3d.detach(),
            anchor_points * stride_tensor,
            (gt_labels, gt_bboxes, gt_center_2d, gt_size_2d, gt_center_3d, gt_size_3d, gt_depth, gt_heading_bin, gt_heading_res),
            mask_gt,
            stride_tensor,
            calibs,
            mean_sizes
        )
        try:
            (_, target_scores, target_center_2d, target_size_2d, target_center_3d,
             target_size_3d, target_depth, target_heading_bin, target_heading_res) = targets
        except Exception as e:
            return loss.sum() * batch_size, loss

        target_scores_sum = max(target_scores.sum(), 1)

        targets_2d = targets[2:4]
        targets_3d = targets[4:9] # center, size, depth, head_bin, head_res

        #self.plot_assignments(batch, targets_2d, fg_mask, pred_bboxes, stride_tensor, targets_3d,  pred_kps, gt_kps, mask_gt)

        loss[0] = (self.compute_box2d_loss(targets_2d, pred_2d, anchor_points, stride_tensor, fg_mask, target_scores_sum)
                   * self.hyp.loss2d)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum * self.hyp.cls

        loss[2:6] = self.compute_box3d_loss(targets_3d, pred_3d, anchor_points, stride_tensor,
                                            fg_mask, target_scores_sum)

        if self.hyp.distillation:
            embeddings = torch.cat([emb.view(feats[0].shape[0], 128, -1) for emb in embeddings], dim=2)
            loss[6] = self.supervisor.forward(
                batch["img"].detach(), gt_center_3d, embeddings, fg_mask.bool(),
                target_gt_idx, mask_gt.bool().squeeze(-1), batch["mixed"].bool()
            ) / target_scores_sum

        return loss.sum() * batch_size, loss

    def plot_assignments(self, batch, targets_2d, fg_mask, pred_bboxes, stride_tensor, targets_3d,  pred_kps, gt_kps, mask_gt):
        self.debug_show_assigned_targets2d(batch, targets_2d, fg_mask, pred_bboxes, stride_tensor)
        self.debug_show_assigned_targets3d(batch, targets_3d, fg_mask, pred_kps, gt_kps, mask_gt)
        self.debug_show_pred_bevs(pred_kps, gt_kps, fg_mask, mask_gt, stride_tensor)

    def compute_loss_weights(self, current_loss):
        weights = torch.ones(6, device=self.device)
        if current_loss[1] > 1.9:
            weights[2:] = 0
        return weights

    def compute_box2d_loss(self, targets_2d, pred_2d, anchor_points, stride_tensor, fg_mask, num_targets):
        target_center_2d, target_size_2d = targets_2d
        pred_2d = pred_2d * stride_tensor
        anchor_points = anchor_points * stride_tensor

        pred_offset = pred_2d[fg_mask][..., :2]
        pred_size = pred_2d[fg_mask][..., 2:]
        target_size = target_size_2d[fg_mask]
        target_offset = (target_center_2d - anchor_points)[fg_mask]

        offset2d_loss = F.l1_loss(pred_offset, target_offset, reduction="mean")
        size2d_loss = F.l1_loss(pred_size, target_size, reduction="mean")

        return (size2d_loss + offset2d_loss) / num_targets

    def compute_box3d_loss(self, targets_3d, pred_3d, anchor_points, stride_tensor, fg_mask, num_targets):
        pred_depth = pred_3d[fg_mask][..., -2]
        pred_depth_un = pred_3d[fg_mask][..., -1]
        target_depth = targets_3d[-3][fg_mask].squeeze()
        depth_loss = (laplacian_aleatoric_uncertainty_loss_new(pred_depth, target_depth, pred_depth_un).sum()
                      / num_targets * self.hyp.depth)

        anchor_points = anchor_points * stride_tensor
        pred_offset = (pred_3d[..., :2] * stride_tensor)[fg_mask]
        target_center_3d = targets_3d[0]
        target_offset = (target_center_3d - anchor_points)[fg_mask]
        offset3d_loss = (F.l1_loss(pred_offset, target_offset, reduction="mean")
                         / num_targets * self.hyp.offset3d)

        pred_size = pred_3d[fg_mask][..., 2:5]
        target_size = targets_3d[1][fg_mask]
        size3d_loss = (F
                       .l1_loss(pred_size, target_size, reduction="sum")
                       / num_targets * self.hyp.size3d)

        pred_heading = pred_3d[fg_mask][..., 5:29]
        target_bin = targets_3d[-2][fg_mask]
        target_res = targets_3d[-1][fg_mask]
        heading_loss = (compute_heading_loss(pred_heading, target_bin, target_res)
                        / num_targets * self.hyp.heading)

        if depth_loss != depth_loss:
            print('badNAN----------------depth_loss', depth_loss)
        if offset3d_loss != offset3d_loss:
            print('badNAN----------------offset3d_loss', offset3d_loss)
        if size3d_loss != size3d_loss:
            print('badNAN----------------size3d_loss', size3d_loss)
        if heading_loss != heading_loss:
            print('badNAN----------------heading_loss', heading_loss)

        return torch.stack((depth_loss, offset3d_loss, size3d_loss, heading_loss))

    def debug_show_assigned_targets2d(self, batch, targets_2d, fg_mask, pred_bboxes, stride_tensor):
        target_center_2d, target_size_2d = targets_2d
        target_bboxes = torch.cat(
            (target_center_2d - target_size_2d / 2, target_center_2d + target_size_2d / 2), dim=-1)

        color = {8: (255, 255, 0), 16: (0, 255, 255), 32: (255, 0, 255)}
        mean = 0
        std = 1
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'wspace': 0, 'hspace': 0},
                                 constrained_layout=True)
        for i, ax in enumerate(np.ravel(axes)):
            img = batch["img"][i].detach().cpu().numpy().transpose(1, 2, 0).copy()
            img = np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)

            target_boxes = target_bboxes[i][fg_mask[i]].int().cpu()
            for box in target_boxes:
                p1, p2 = box.split((2, 2), dim=0)
                cv2.rectangle(img, p1.numpy(), p2.numpy(), (255, 0, 0), 3)  # gt

            pred_boxes = pred_bboxes[i][fg_mask[i]].cpu()
            stride = stride_tensor[fg_mask[i]].cpu()
            for j, box in enumerate(pred_boxes):
                c = color[stride[j].item()]
                p1, p2 = box.split((2, 2), dim=0)
                cv2.rectangle(img, p1.int().numpy(), p2.int().numpy(), c)  # gt
                cv2.circle(img, (p1 + (p2 - p1) / 2).int().numpy(), 4, (0, 255, 255), -1)

            t_center_2d = target_center_2d[i][fg_mask[i]].int().cpu()
            for center2d in t_center_2d:
                cv2.circle(img, center2d.numpy(), 5, (0, 0, 255), -1)

            ax.imshow(img)
            ax.axis("off")
        plt.show()
        print()

    def debug_show_assigned_targets3d(self, batch, targets_3d, fg_mask, pred_kps, gt_kps, mask_gt):
        target_center_3d, _, _, _, _ = targets_3d
        box_idxs = [0, 1, 2, 3, 0,  # base
                    4, 7, 3, 7,  # right
                    6, 2, 6,  # back
                    5,  # left
                    4, 1, 5, 0]  # front
        color_none_bottom, color_bottom = ((255, 0, 0), (255, 0, 0))
        color_gt_none_bot, color_gt_bot = ((0, 255, 0), (0, 0, 255))
        fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'wspace': 0, 'hspace': 0},
                                 constrained_layout=True)
        for i, ax in enumerate(np.ravel(axes)):
            img = batch["img"][i].detach().cpu().numpy().transpose(1, 2, 0).copy()
            img = np.clip((img) * 255, 0, 255).astype(np.uint8)

            t_center_3d = target_center_3d[i][fg_mask[i]].int().cpu()
            pd_kps = self.project_to_image(pred_kps[i][fg_mask[i]], batch["calib"][i]).int().cpu()
            gr_kps = self.project_to_image(gt_kps[i][mask_gt[i].bool().squeeze(-1)], batch["calib"][i]).int().cpu()
            for center_3d in t_center_3d:
                cv2.circle(img, center_3d.numpy(), 5, (0, 0, 255), -1)
            for kp in pd_kps:
                cv2.polylines(img, [kp[box_idxs[:5]].numpy().astype(np.int32).reshape((-1, 1, 2))], isClosed=False,
                              color=color_none_bottom, thickness=1, lineType=cv2.LINE_AA)
                cv2.polylines(img, [kp[box_idxs[5:]].numpy().astype(np.int32).reshape((-1, 1, 2))], isClosed=True,
                              color=color_bottom, thickness=1, lineType=cv2.LINE_AA)
            for kp in gr_kps:
                cv2.polylines(img, [kp[box_idxs[:5]].numpy().astype(np.int32).reshape((-1, 1, 2))], isClosed=False,
                              color=color_gt_none_bot, thickness=1, lineType=cv2.LINE_AA)
                cv2.polylines(img, [kp[box_idxs[5:]].numpy().astype(np.int32).reshape((-1, 1, 2))], isClosed=True,
                              color=color_gt_bot, thickness=1, lineType=cv2.LINE_AA)

            ax.imshow(img)
            ax.axis("off")
        plt.show()
        print()

    def debug_show_pred_bevs(self, pred_kps, gt_kps, fg_mask, mask_gt, stride_tensor):
        max_imgs = 4
        fig, ax = plt.subplots(math.ceil(max_imgs ** 0.5), math.ceil(max_imgs ** 0.5),
                               figsize=(36, 18), gridspec_kw={'wspace': 0, 'hspace': 0}, constrained_layout=True)
        ax = ax.ravel()
        color = {8: (255, 255, 0), 16: (0, 255, 255), 32: (255, 0, 255)}

        for i, anchors in enumerate(pred_kps):
            if i >= max_imgs:
                break
            MAX_DIST = 70
            SCALE = 30

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

            for j, anchor in enumerate(anchors[torch.logical_not(fg_mask[i])].cpu().numpy()):
                c = color[stride_tensor[j].item()]
                bottom_corners = (anchor[:4] * SCALE)
                x = bottom_corners[:, 0] + R
                y = -bottom_corners[:, 2] + R
                pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1).astype(np.int32)[[0, 1, 3, 2]]
                space = cv2.polylines(space, pts=[pts], isClosed=True, color=c)
            for assigned in anchors[fg_mask[i]].cpu().numpy():
                bottom_corners = (assigned[:4] * SCALE)
                x = bottom_corners[:, 0] + R
                y = -bottom_corners[:, 2] + R
                pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1).astype(np.int32)[
                    [0, 1, 3, 2]]
                space = cv2.polylines(space, pts=[pts], isClosed=True, color=(0, 0, 255), thickness=2)
            for gt in gt_kps[i][mask_gt[i].bool().squeeze(-1)].cpu().numpy():
                bottom_corners = (gt[:4] * SCALE)
                x = bottom_corners[:, 0] + R
                y = -bottom_corners[:, 2] + R
                pts = np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), axis=1).astype(np.int32)[
                    [0, 1, 3, 2]]
                space = cv2.polylines(space, pts=[pts], isClosed=True, color=(0, 255, 0), thickness=2)

            ax[i].imshow(space)
            ax[i].axis("off")
        plt.show()
        print()

    def project_to_image(self, kps, calib):
        sample_num = kps.shape[0]
        corners3d_hom = torch.cat((kps, torch.ones((sample_num, 8, 1), device=calib.device)), axis=2)  # (N, 8, 4)

        mat = self.get_mat(calib)
        img_pts = torch.matmul(corners3d_hom, mat.T.double()) # (N, 8, 3)
        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        boxes_corner = torch.cat((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)
        return boxes_corner

    def get_mat(self, calib):
        mat = torch.eye(4, device=calib.device)[:3]
        cu, cv, fu, fv, tx, ty = calib.split((1, 1, 1, 1, 1, 1), dim=-1)
        mat[0, 2] = cu
        mat[1, 2] = cv
        mat[0, 0] = fu
        mat[1, 1] = fv
        mat[0, 3] = tx * (-fu)
        mat[1, 3] = ty * (-fv)
        return mat

def laplacian_aleatoric_uncertainty_loss_new(input, target, log_variance):
    '''
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    '''
    loss = 1.4142 * torch.exp(-0.5*log_variance) * torch.abs(input - target) + 0.5*log_variance
    return loss


def compute_heading_loss(input, target_cls, target_reg):
    target_cls = target_cls.view(-1).long()
    target_reg = target_reg.view(-1)

    # classification loss
    input_cls = input[..., 0:12]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='sum')

    # regression loss
    input_reg = input[..., 12:24]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='sum')

    return cls_loss + reg_loss

class SupervisionLoss:
    def __init__(self, model):
        from ultralytics.utils.dino import DinoDepther
        self.device = next(model.parameters()).device  # get model device
        self.args = model.args
        self.model = model
        self.foundation_model = DinoDepther()
        self.foundation_model.load(self.args.dino_path)
        self.T = self.args.distillation_temp
        self.weight = self.args.distillation_weight
        self.criterion = self.args.distillation_loss
        self.no_mixup = self.args.distillation_no_mixup
        if self.criterion == "cos":
            self.loss = nn.CosineEmbeddingLoss()
        elif self.criterion == "mse":
            self.loss = nn.MSELoss()

    def forward(self, imgs, gt_center_3d, pred_embeddings, fg_mask, target_gt_idx, mask_gt, mixed_mask):
        loss = torch.zeros(imgs.shape[0], device=imgs.device)

        with torch.no_grad():
            depth_maps, dino_embeddings = self.foundation_model(imgs)
        #self.plot_depth_maps(depth_maps, imgs)

        for batch_idx in range(depth_maps.shape[0]):
            if mask_gt[batch_idx].any() and (not self.no_mixup or not mixed_mask[batch_idx]):
                img_size = torch.tensor(imgs.shape[2:][::-1], device=gt_center_3d.device)
                dino_embed_size = torch.tensor(dino_embeddings.shape[2:][::-1], device=gt_center_3d.device)
                center3d = gt_center_3d[batch_idx][mask_gt[batch_idx]] / img_size * dino_embed_size
                dino_emb = dino_embeddings.transpose(1, 3)[
                        batch_idx,
                        center3d[:, 0].round().long().clamp(min=0, max=dino_embed_size[0] - 1),
                        center3d[:, 1].round().long().clamp(min=0, max=dino_embed_size[1] - 1)
                ][target_gt_idx[batch_idx][fg_mask[batch_idx]]]
                pred_emb = pred_embeddings.transpose(1, 2)[batch_idx][fg_mask[batch_idx]]

                if self.criterion == "soft":
                    soft_targets = nn.functional.softmax(dino_emb / self.T, dim=-1)
                    soft_prob = nn.functional.log_softmax(pred_emb / self.T, dim=-1)
                    loss[batch_idx] = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (
                            self.T ** 2)
                elif self.criterion == "mse":
                    loss[batch_idx] = self.loss(pred_emb, dino_emb)
                elif self.criterion == "cos":
                    loss[batch_idx] = self.loss(pred_emb, dino_emb, target=torch.ones(dino_emb.size(0)).to(dino_emb.device))
                else:
                    raise RuntimeError(f"Unknown criterion function: {self.criterion}")
            else:
                loss[batch_idx] = 0
        return loss.sum() * self.weight


    def plot_depth_maps(self, depth_maps, imgs):
        fig ,axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.tight_layout()
        axes = axes.ravel()
        for i, ax in enumerate(axes):
            dmap = cv2.applyColorMap(((depth_maps[i]/depth_maps[i].max()).cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
            img = (255*imgs[i].cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
            res = np.vstack((dmap, img))
            ax.imshow(res)

        plt.show()


class ForegroundDepthMapLoss(nn.Module):

    def __init__(self,
                 model,
                 alpha=0.25,
                 gamma=2.0,
                 fg_weight=13,
                 bg_weight=1,
                 downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.device = next(model.parameters()).device  # get model device
        self.args = model.args
        self.num_bins = 80
        self.model = model
        self.balancer = Balancer(
            downsample_factor=downsample_factor,
            fg_weight=fg_weight,
            bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = LogitFocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")

    def bin_depths(self, depth_map, depth_min, depth_max, num_bins, target=False):
        mode = "LID"
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        elif mode == "SID":
            indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
                      (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        return indices

    def forward(self, depth_logits, depth_maps):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """

        # downsample depth_maps by 16
        depth_maps = transforms.Resize(size=[depth_maps.shape[1] // 16, depth_maps.shape[2] // 16], interpolation=InterpolationMode.NEAREST)(depth_maps)
        # Bin depth map to create target
        depth_target = self.bin_depths(depth_maps,
                                       target=True,
                                       depth_min=self.args.min_depth_threshold,
                                       depth_max=self.args.max_depth_threshold,
                                       num_bins=self.num_bins)
        # Compute loss
        loss = self.loss_func(depth_logits, depth_target.long())
        # Compute foreground/background balancing
        loss = self.balancer(loss=loss, fg_mask=depth_maps>0)

        return loss


class Balancer(nn.Module):
    def __init__(self, fg_weight, bg_weight, downsample_factor=1):
        """
        Initialize fixed foreground/background loss balancer
        Args:
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.downsample_factor = downsample_factor

    def forward(self, loss, fg_mask):
        """
        Forward pass
        Args:
            loss [torch.Tensor(B, H, W)]: Pixel-wise loss
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Total loss after foreground/background balancing
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        # Compute masks
        bg_mask = ~fg_mask

        # Compute balancing weights
        weights = self.fg_weight * fg_mask + self.bg_weight * bg_mask
        num_pixels = fg_mask.sum() + bg_mask.sum()

        # Compute losses
        loss *= weights
        fg_loss = loss[fg_mask].sum() / num_pixels
        bg_loss = loss[bg_mask].sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        return loss


def compute_fg_mask(gt_boxes2d, shape, num_gt_per_img, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d [torch.Tensor(B, N, 4)]: 2D box labels
        shape [torch.Size or tuple]: Foreground mask desired shape
        downsample_factor [int]: Downsample factor for image
        device [torch.device]: Foreground mask desired device
    Returns:
        fg_mask [torch.Tensor(shape)]: Foreground mask
    """
    #ipdb.set_trace()
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
    gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
    B = len(gt_boxes2d)
    for b in range(B):
        for n in range(gt_boxes2d[b].shape[0]):
            u1, v1, u2, v2 = gt_boxes2d[b][n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def one_hot(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))
    # ipdb.set_trace()
    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)
    # ipdb.set_trace()
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: Optional[float] = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
    Return:
        the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)
    # ipdb.set_trace()
    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))
    # ipdb.set_trace()
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class LogitFocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)

