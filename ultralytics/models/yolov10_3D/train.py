from ultralytics.models.yolo.detect import DetectionTrainer
from .val import YOLOv10_3DDetectionValidator
from .model import YOLOv10_3DDetectionModel
from copy import copy
from ultralytics.utils import RANK
from ultralytics.data.dataset import KITTIDataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.plotting import plot_labels_3D, KITTIVisualizer, plot_images
import numpy as np

class YOLOv10_3DDetectionTrainer(DetectionTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visualizer = KITTIVisualizer()

    def build_dataset(self, img_path, mode="train", batch=None):
        return KITTIDataset(img_path, mode, self.args)

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = ("box_om", "cls_om", "dep_om", "o3d_om", "s3d_om", "hd_om",
                           "box_oo", "cls_oo", "dep_oo", "o3d_oo", "s3d_oo", "hd_oo")
        return YOLOv10_3DDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = YOLOv10_3DDetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        batch["calib"] = batch["calib"].to(self.device)
        batch["coord_range"] = batch["coord_range"].to(self.device)
        batch["img"] = batch["img"].to(self.device)
        return batch

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        labels = self.train_loader.dataset.labels
        plot_labels_3D(labels, class2id=self.train_loader.dataset.cls2id, save_dir=self.save_dir, on_plot=self.on_plot)

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"].clone(),
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
            max_subplots=9
        )
        self.visualizer.plot_batch(batch, self.train_loader.dataset, self.save_dir / f"train_batch3d{ni}.jpg")
