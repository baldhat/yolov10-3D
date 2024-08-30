from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.data.dataset import KITTIDataset
from ultralytics.utils.plotting import KITTIVisualizer
from ultralytics.data.decode_helper import decode_batch, decode_preds
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.eval import eval_from_scrach

import torch
import numpy as np
from pathlib import Path
import os
from sklearn.neighbors import KernelDensity



class YOLOv10_3DDetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco
        self.visualizer = KITTIVisualizer()
        self.results = {}

    def build_dataset(self, img_path, mode="val", batch=None):
        return KITTIDataset(img_path, mode, self.args)

    def postprocess(self, preds):
        if isinstance(preds, dict):
            predsO = preds["one2one"]
            predsM = preds["one2many"]

        if isinstance(predsO, (list, tuple)):
            predsO = predsO[0]
        
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
                    depths = depths[mask]
                    weights = depth_scores / depth_scores.sum()
                    kde = KernelDensity(bandwidth="silverman", kernel='gaussian').fit(depths.unsqueeze(-1).cpu(), sample_weight=weights.cpu())
                    logprob = torch.tensor(kde.score_samples(depths.unsqueeze(-1).cpu()))
                    max_ind = torch.argmax(logprob)
                    predsO[i, j, -4] = depths[max_ind]
                    predsO[i, j, -3] = depth_uncerts[max_ind]
        return predsO # lala
        

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float())
        for k in ["batch_idx", "bboxes", "cls", "depth", "center_3d", "center_2d", "size_2d", "heading_bin",
                  "heading_res", "size_3d"]:
            batch[k] = batch[k].to(self.device)
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

        self.save_results(self.results, output_dir=self.save_dir)
        self.results = {}
        try:
            result = eval_from_scrach(
                self.dataloader.dataset.label_dir,
                os.path.join(self.save_dir, 'preds'),
                ap_mode=40)
            self.metrics.car3d = result
            self.metrics.carAP3d = result["3d@0.70"][1]
        except:
            print("Failed to evaluate mAP")
        return self.metrics.results_dict

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'preds')
        os.makedirs(output_dir, exist_ok=True)
        for img_file in results.keys():
            out_path = os.path.join(output_dir, img_file)
            f = open(out_path, 'w')
            for i in range(len(results[img_file])):
                class_name = self.dataloader.dataset.class_name[int(results[img_file][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_file][i])):
                    f.write(' {:.2f}'.format(results[img_file][i][j]))
                f.write('\n')
            f.close()



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
        return decode_batch(batch, calibs, self.dataloader.dataset.cls_mean_size, use_camera_dis=self.dataloader.dataset.use_camera_dis)

    def _prepare_preds(self, preds, batch):
        infos_ = self.collate_infos(batch)
        calibs = [self.dataloader.dataset.get_calib(info) for info in infos_['img_id']]
        inv_trans = [inv for inv in infos_["trans_inv"]]
        return decode_preds(preds, calibs, self.dataloader.dataset.cls_mean_size, batch["im_file"], inv_trans,
                            use_camera_dis=self.dataloader.dataset.use_camera_dis)

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
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred_bev.jpg",
            names=self.names
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