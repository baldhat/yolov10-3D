from ultralytics.data.datasets.rope3d import Rope3Dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.data.datasets.kitti import KITTIDataset
from ultralytics.data.datasets.waymo import WaymoDataset
from ultralytics.data.datasets.omni3d import Omni3Dataset
from ultralytics.utils.plotting import KITTIVisualizer
from ultralytics.utils.metrics import box_iou

import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import KernelDensity

class YOLOv10_3DDetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco
        self.visualizer = KITTIVisualizer()
        self.results = {}
        self.dino_model = None

    def build_dataset(self, img_path, mode="val", batch=None):
        dataset_yaml = self.args.data.split("/")[-1]
        if dataset_yaml == "kitti.yaml":
            return KITTIDataset(img_path, mode, self.args)
        elif dataset_yaml == "waymo.yaml":
            return WaymoDataset(img_path, mode, self.args)
        elif dataset_yaml == "omni3d.yaml" or dataset_yaml == "cdrone.yaml":
            return Omni3Dataset(img_path, mode, self.args)
        elif dataset_yaml == "rope3d.yaml":
            return Rope3Dataset(img_path, mode, self.args)
        else:
            raise NotImplemented("Yolov10_3D only supports Kitti and Waymo datasets")

    def postprocess(self, preds):
        if isinstance(preds, dict):
            predsO = preds["one2one"]

        if isinstance(predsO, (list, tuple)):
            predsO = predsO[0]
        
        if self.args.use_o2m_depth:
            predsM = preds["one2many"]
            if isinstance(predsM, (list, tuple)):
                predsM = predsM[0]
        
        predsO = predsO.transpose(-1, -2)
        regO, scoresO, labelsO = ops.v10_3Dpostprocess(predsO, self.args.max_det, self.nc)
        preds = torch.cat((regO, scoresO.unsqueeze(-1), labelsO.unsqueeze(-1)), dim=-1)
        
        if self.args.use_o2m_depth:
            predsM = predsM.transpose(-1, -2)
            regM, scoresM, labelsM = ops.v10_3Dpostprocess(predsM, self.args.max_det * 5, self.nc)
            predsM = torch.cat((regM, scoresM.unsqueeze(-1), labelsM.unsqueeze(-1)), dim=-1)
            
            preds = self.aggregate_o2m_preds(preds, predsM)
            
        elif self.args.use_dino_depth:
            preds = self.dino_depth_pred(preds)
        
        return preds

    def dino_depth_pred(self, preds):
        if self.dino_model is None:
            from ultralytics.utils.dino import DinoDepther
            self.dino_model = DinoDepther()
            self.dino_model.load(self.args.dino_path)
        with torch.inference_mode():
            imgs = self.batch["img"]
            depth_maps, embeddings = self.dino_model.forward(imgs)
            pred_center3d = preds[..., 4:6]
            for batch_idx, center3d in enumerate(pred_center3d):
                pred_deps = depth_maps[
                    batch_idx,
                    pred_center3d[batch_idx, :, 1].long().clamp(min=0, max=depth_maps.shape[1] - 1),
                    pred_center3d[batch_idx, :, 0].long().clamp(min=0, max=depth_maps.shape[2] - 1)]
                preds[batch_idx, :, -4] = pred_deps
        return preds

    def aggregate_o2m_preds(self, predsO, predsM, thres=0.1):
        # bbox, pred_center3d, pred_s3d, pred_hd, pred_dep, pred_dep_un, scores, labels = preds.split((4, 2, 3, 24, 1, 1, 1, 1), dim=-1)
        bboxO = predsO[:, :, 0:4] # format: xyxy
        bboxM = predsM[:, :, 0:4]
        
        for i in range(predsO.shape[0]):
            iou = box_iou(bboxO[i], bboxM[i])
            for j in range(predsO.shape[1]):
                matches = iou[j] > 0.9
                depths = torch.cat((predsO[i, j, -4].unsqueeze(0), predsM[i, matches, -4]))
                depth_uncerts = torch.cat((predsO[i, j, -3].unsqueeze(0), predsM[i, matches, -3]))
                cls = torch.cat((predsO[i, j, -1].unsqueeze(0), predsM[i, matches, -1]))
                depth_scores = torch.exp(-depth_uncerts)
                mask = torch.logical_and(depth_scores > thres, cls == predsO[i, j, -1])
                if mask.nonzero().numel() > 1:
                    depth_scores = depth_scores[mask]
                    depths = depths[mask].cpu()
                    weights = depth_scores / depth_scores.sum()
                    kde = KernelDensity(bandwidth="silverman", kernel='gaussian').fit(depths.unsqueeze(-1), sample_weight=weights.cpu())
                    proposals = np.expand_dims(np.linspace(depths.min(), depths.max(), 500), -1)
                    logprob = torch.tensor(kde.score_samples(proposals))
                    max_ind = torch.argmax(logprob)
                    predsO[i, j, -4] = torch.tensor(proposals[max_ind])#depths[max_ind]
                    predsO[i, j, -3] = depth_uncerts[0]#logprob[max_ind]#depth_uncerts[max_ind]
        return predsO # lala
        

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float())
        for k in ["batch_idx", "bboxes", "cls", "depth", "center_3d", "center_2d", "size_2d", "heading_bin",
                  "heading_res", "size_3d", "calib", "rot_mat"]:
            batch[k] = batch[k].to(self.device)
        self.batch = batch
        return batch

    def update_metrics(self, preds, batch):
        preds = self._prepare_preds(preds, batch)
        targets = self._prepare_batch(batch)
        self.results.update(preds)

        for si, (pred, target) in enumerate(zip(preds.values(), targets.values())):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pred = torch.tensor(np.array(pred), device=self.device)
            target = torch.tensor(np.array(target), device=self.device)
            cls = target[:, 0] if target.numel() > 0 else torch.empty(0, device=self.device)
            bbox = target[:, 2:6] if target.numel() > 0 else torch.empty(0, device=self.device)
            nl = len(cls) if type(cls) is list else cls.numel()
            stat["target_cls"] = cls
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 0] = 0

            pred_conf = pred[:, -1]
            pred_cls = pred[:, 0]
            pred_bbox = pred[:, 2:6]
            stat["conf"] = pred_conf
            stat["pred_cls"] = pred_cls

            pred2d = torch.cat((pred_bbox, pred_conf.unsqueeze(-1), pred_cls.unsqueeze(-1)), dim=-1)
            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(pred2d, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(pred2d, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(pred2d, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(pred2d, self.args.save_conf, batch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        self.nt_per_class = np.bincount(
            stats["target_cls"].astype(int), minlength=self.nc
        )  # number of targets per class


        try:
            self.metrics.metric3d = self.dataloader.dataset.get_stats(self.results, self.save_dir)
            self.results = {}
        except Exception as e:
            print(f"Failed to evaluate mAP: {e}")
        return self.metrics.results_dict

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, batch):
        infos_ = self.collate_infos(batch)
        calibs = [self.dataloader.dataset.get_calib(info) for info in infos_['img_id']]
        return self.dataloader.dataset.decode_batch_eval(batch, calibs)

    def _prepare_preds(self, preds, batch):
        infos_ = self.collate_infos(batch)
        calibs = [self.dataloader.dataset.get_calib(info) for info in infos_['img_id']]
        inv_trans = [inv for inv in infos_["trans_inv"]]
        return self.dataloader.dataset.decode_preds_eval(preds, calibs, batch["im_file"], batch["ratio_pad"], inv_trans)

    def plot_predictions(self, batch, preds, ni):
        self.visualizer.plot_preds(
            batch,
            preds,
            self.dataloader.dataset,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred3d.jpg",
            names=self.names
        )
        self.visualizer.plot_bev(
            batch,
            preds,
            self.dataloader.dataset,
            fname=self.save_dir / f"val_batch{ni}_pred_bev.jpg"
        )

    def plot_val_samples(self, batch, ni):
        self.visualizer.plot_batch(
            batch,
            self.dataloader.dataset,
            self.save_dir / f"val_batch{ni}_label3d.jpg",
        )

    @staticmethod
    def collate_infos(batch):
        """Collates data samples into batches."""
        infos = {}
        keys = batch["info"][0].keys()
        values = list(zip(*[list(b.values()) for b in batch["info"]]))
        for i, k in enumerate(keys):
            infos[k] = np.stack(values[i], 0)
        return infos