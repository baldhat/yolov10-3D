import sys
import os
import random
import numpy as np
import tqdm
from pathlib import Path

REPO_PATH = "/home/stud/mijo/dev/dinov2" # Specify a local path to the repository (or use installed package instead)
sys.path.append(REPO_PATH)

import torch
import mmcv
from mmcv.runner import load_checkpoint
from torchvision import transforms
import urllib
import math
import itertools
from functools import partial
import torch.nn.functional as F

import warnings
from dinov2.eval.depth.models import build_depther


def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther



def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()



class DinoDepther(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self.load_backbone()
        self.head = self.load_head()

    def load_head(self):
        HEAD_TYPE = "linear"
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        head_config_url = f"{DINOV2_BASE_URL}/{self.backbone_name}/{self.backbone_name}_kitti_{HEAD_TYPE}_config.py"
        #head_checkpoint_url = f"{DINOV2_BASE_URL}/{self.backbone_name}/{self.backbone_name}_kitti_{HEAD_TYPE}_head.pth"

        cfg_str = load_config_from_url(head_config_url)
        cfg_str = cfg_str.replace("BNHead", "ConvHead")
        cfg_str = cfg_str.replace("classify=True", "classify=False")
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

        model = create_depther(
            cfg,
            backbone_model=self.backbone
        )

        #load_checkpoint(model, head_checkpoint_url, map_location="cpu")
        model.eval()
        model.cuda()
        return model

    def load_backbone(self):
        BACKBONE_SIZE = "small"  # in ("small", "base", "large" or "giant")
        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        backbone_arch = backbone_archs[BACKBONE_SIZE]
        self.backbone_name = f"dinov2_{backbone_arch}"
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        backbone_model.eval()
        backbone_model.cuda()
        return backbone_model

    def transform_imgs(self, imgs):
        self.img_size = [imgs.shape[2], imgs.shape[3]]
        t = transforms.Compose([
            lambda x: 255.0 * x,  # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
            transforms.Resize(size=[imgs.shape[2] - imgs.shape[2] % 14, imgs.shape[3] - imgs.shape[3] % 14])
        ])
        return t(imgs)

    def transform_back(self, depth_maps):
        t = transforms.Compose([
            transforms.Resize(size=self.img_size)
        ])
        return t(depth_maps)

    def forward(self, x):
        input = self.transform_imgs(x)
        features = self.head.backbone(input)
        depth_maps, embeddings = self.head.decode_head.forward_test(features, None, None)  # Add return embeddings
        depth_maps = self.transform_back(depth_maps).squeeze(1)
        return depth_maps, embeddings

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


from ultralytics.data.datasets.kitti import KITTIDataset
from ultralytics.data.build import InfiniteDataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(dataset, batch, workers, shuffle=True):
    """Return an InfiniteDataLoader or DataLoader for training or validation set."""
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), workers])  # number of workers
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 )
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=False,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def preprocess_gts(targets, batch_size):
    i = targets[:, 0]  # image index
    _, counts = i.unique(return_counts=True)
    counts = counts.to(dtype=torch.int32)
    out = torch.zeros(batch_size, counts.max(), 3).cuda()
    for j in range(batch_size):
        matches = i == j
        n = matches.sum()
        if n:
            out[j, :n] = targets[matches, 1:]
    return out

def get_loss(output, batch):
    batch_size = output.shape[0]
    gt_batchid = batch["batch_idx"]
    gt_depths = batch["depth"]
    gt_center_3ds = batch["center_3d"]
    gts = torch.cat((gt_batchid.view(-1, 1), gt_depths.view(-1, 1), gt_center_3ds), dim=1)
    gt_depths, gt_center_3ds = preprocess_gts(gts, batch_size).split((1, 2), dim=2)
    mask_gt = gt_center_3ds.sum(2, keepdim=True).gt_(0).bool().squeeze(-1)
    loss = torch.zeros(batch_size, requires_grad=True).cuda().double()
    for batch_idx, pred in enumerate(output):
        num_objects = mask_gt[batch_idx].sum()
        if num_objects.item() > 0:
            pred_deps = output[batch_idx, gt_center_3ds[batch_idx, :, 1].long(), gt_center_3ds[batch_idx, :, 0].long()].unsqueeze(-1)
            loss[batch_idx] = torch.nn.functional.smooth_l1_loss(pred_deps.double()[mask_gt[batch_idx]], gt_depths[batch_idx].double()[mask_gt[batch_idx]], reduction="sum") / num_objects
    return loss.sum() / batch_size

def validate(epoch_index, model, dataloader):
    model.eval()
    mean_loss = 0
    with torch.inference_mode():
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in pbar:
            pred, feats = model(batch["img"].cuda())
            loss = get_loss(pred, batch)
            mean_loss += loss.item()
            pbar.set_description(f"Eval Epoch {epoch_index}: MSE: {loss.item()}")
    print(f"Mean loss: {mean_loss / len(dataloader)}")
    return mean_loss / len(dataloader)

def train_one_epoch(epoch_index, model, dataloader, optimizer):
    model.train()
    mean_loss = 0
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for i, batch in pbar:
        optimizer.zero_grad()
        pred, feats = model(batch["img"].cuda())
        loss = get_loss(pred, batch)
        mean_loss += loss.item()
        pbar.set_description(f"Epoch {epoch_index}: MSE: {loss.item()}")
        loss.backward()
        optimizer.step()
    print(f"Mean loss: {mean_loss / len(dataloader)}")
    return mean_loss / len(dataloader)

class Args:
    cam_dis = False
    fliplr = 0.5
    random_crop = 0.5
    scale = 0.4
    min_scale = 0.6
    max_scale = 1.4
    translate = 0.1
    mixup = 0.5
    max_depth_threshold = 70
    min_depth_threshold = 0.5
    seed = 1
    

def main(save_dir):
    train_file_path = "/home/stud/mijo/dev/datasets/KITTI/ImageSets/train.txt"
    val_file_path = "/home/stud/mijo/dev/datasets/KITTI/ImageSets/val.txt"
    args = Args()
    train_dataset = KITTIDataset(train_file_path, "train", args)
    val_dataset = KITTIDataset(val_file_path, "val", args)
    train_dataloader = build_dataloader(train_dataset, 64, 8, shuffle=True)
    val_dataloader = build_dataloader(val_dataset, 64, 8, shuffle=False)

    model = DinoDepther()
    model.train()

    freeze_backbone(model)

    optimizer = torch.optim.Adam(model.head.parameters(), lr=6e-5)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.01, total_iters=100)

    best_eval_loss = 100000
    best_epoch = 0
    train_losses = []
    val_losses = []

    for epoch in range(100):
        train_loss = train_one_epoch(epoch, model, train_dataloader, optimizer)
        eval_loss = validate(epoch, model, val_dataloader)

        train_losses.append(train_loss)
        val_losses.append(eval_loss)
        
        Path(os.path.join(save_dir, "dino")).mkdir(parents=True, exist_ok=True)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            model.save(os.path.join(save_dir, "dino/best.pt"))
        lr_scheduler.step()
        print(f"Best epoch: {best_epoch}")
        np.save(os.path.join(save_dir, "dino/losses"), np.array((train_losses, val_losses)))


if __name__ == '__main__':
    save_dir = sys.argv[1]
    print(save_dir)
    main(save_dir)