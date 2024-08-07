from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch
from ultralytics.data.dataset import KITTIDataset

class YOLOv10_3DDetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def build_dataset(self, img_path, mode="val", batch=None):
        return KITTIDataset(img_path, mode, self.args)
    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Acknowledgement: Thanks to sanha9999 in #190 and #181!
        if preds.shape[-1] == 6:
            return preds
        else:
            preds = preds.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
            bboxes = ops.xywh2xyxy(boxes)
            return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

    def preprocess(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float())
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)
        return batch
