from ultralytics.models.yolo.detect import DetectionTrainer
from .val import YOLOv10_3DDetectionValidator
from .model import YOLOv10_3DDetectionModel
from ultralytics.models.yolov10 import YOLOv10
from ultralytics.nn.modules.head import v10Detect3d
from copy import copy
from ultralytics.data.datasets.kitti import KITTIDataset
from ultralytics.data.datasets.waymo import WaymoDataset
from ultralytics.utils.plotting import plot_labels_3D, KITTIVisualizer, plot_images, plot_training_depth_dist
from torchvision.models import resnet50, ResNet50_Weights

from ...data.datasets.omni3d import Omni3Dataset

import torch


class YOLOv10_3DDetectionTrainer(DetectionTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visualizer = KITTIVisualizer()

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset_yaml = self.args.data.split("/")[-1]
        if dataset_yaml == "kitti.yaml":
            return KITTIDataset(img_path, mode, self.args)
        elif dataset_yaml == "waymo.yaml":
            return WaymoDataset(img_path, mode, self.args)
        elif dataset_yaml == "omni3d.yaml":
            return Omni3Dataset(img_path, mode, self.args)
        else:
            raise NotImplemented("Yolov10_3D only support Kitti and Waymo datasets")

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = [["box_om", "cls_om", "dep_om", "o3d_om", "s3d_om", "hd_om"],
                           ["box_oo", "cls_oo", "dep_oo", "o3d_oo", "s3d_oo", "hd_oo"]]
        if self.args.distillation:
            self.loss_names[0] += ["dis_om"]
            self.loss_names[1] += ["dis_oo"]

        self.loss_names = self.loss_names[0] + self.loss_names[1]

        if self.args.fgdm_loss:
            self.loss_names += ["fgdm"]
            if self.args.fgdm_supervision:
                self.loss_names += ["fgdm_sup"]

        return YOLOv10_3DDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        from copy import deepcopy
        model = YOLOv10_3DDetectionModel(cfg)
        if weights:
            model.load(weights)
        else:
            backbone = YOLOv10.from_pretrained("jameslahm/" + self.model.split("_")[0])
            backbone.model.model[12].f = [-1, 9]
            backbone.model.model[15].f = [-1, 8]
            b = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            t1 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 320, 1), 
                torch.nn.BatchNorm2d(320),
                torch.nn.ReLU()
            )
            t2 = torch.nn.Sequential(
                torch.nn.Conv2d(1024, 640, 1), 
                torch.nn.BatchNorm2d(640),
                torch.nn.ReLU()
            )
            t3 = torch.nn.Sequential(
                torch.nn.Conv2d(2048, 640, 1), 
                torch.nn.BatchNorm2d(640),
                torch.nn.ReLU()
            )

            for i, module in enumerate([b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3,b.layer4, t1, t2, t3]):
                module.i = i
                module.f = -1
            t1.f = 5
            t2.f = 6
            t3.f = 7
            model.model = torch.nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool, b.layer1, b.layer2, b.layer3,b.layer4, t1, t2, t3, *backbone.model.model[11:-1], model.model[-1])
            model.model[-1].stride = model.model[-1].stride[:2]
            return model 

            # model_seq = deepcopy(model.model)
            # for i, module in enumerate(model_seq):
            #     if not isinstance(module, v10Detect3d):
            #         model.model[i] = deepcopy(backbone.model.model[i])
        return model

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        batch["calib"] = batch["calib"].to(self.device)
        batch["img"] = batch["img"].to(self.device)
        batch["non_mix_imgs"] = batch["non_mix_imgs"].to(self.device, non_blocking=True)
        batch["non_mix_imgs"] = (batch["non_mix_imgs"].half() if self.args.half else batch["non_mix_imgs"].float())
        return batch

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        labels = self.train_loader.dataset.labels
        plot_labels_3D(labels, class2id=self.train_loader.dataset.cls2train_id, save_dir=self.save_dir, on_plot=self.on_plot)
        plot_training_depth_dist(self.train_loader.dataset, save_dir=self.save_dir)

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
