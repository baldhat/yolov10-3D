# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from .block import DFL, Proto, ContrastiveHead, BNContrastiveHead
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init
import copy
from ultralytics.utils import ops
from ultralytics.utils.kfe import KFE
import numpy as np

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder"

from ...utils.kprefiner import KeypointRefiner
from ...utils.query_embedder import QueryEmbedder


class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        # number of bounding boxes to predict:
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # bounding box predictor:
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        # class predictor:
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def inference(self, x):
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        y = self.forward_feat(x, self.cv2, self.cv3)
        
        if self.training:
            return y

        return self.inference(y)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        if self.export:
            return dist2bbox(bboxes, anchors, xywh=False, dim=1)
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

class v10Detect(Detect):

    max_det = 300

    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)), \
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), \
                                                nn.Conv2d(c3, self.nc, 1)) for i, x in enumerate(ch))

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
    
    def forward(self, x):
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        if not self.export:
            one2many = super().forward(x)

        if not self.training:
            one2one = self.inference(one2one)
            if not self.export:
                return {"one2many": one2many, "one2one": one2one}
            else:
                assert(self.max_det != -1)
                boxes, scores, labels = ops.v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)
        else:
            return {"one2many": one2many, "one2one": one2one}

    def bias_init(self):
        super().bias_init()
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class v10Detect3d(nn.Module):
    max_det = 50

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=(), dsconv=False, channels=None, use_predecessors=False, detach_predecessors=True,
                 deform=False, common_head=False, num_scales=3, half_channels=False, fgdm_predictor=False,
                 kernel_size_1=3, kernel_size_2=3):
        super().__init__()
        assert (channels is not None)
        self.nc = nc  # number of classes
        self.dsconv = dsconv
        self.nl = num_scales
        self.half_channels = half_channels
        self.output_channels = {
            "cls": self.nc,
            "o2d": 2,
            "s2d": 2,
            "o3d": 2,
            "s3d": 3,
            "hd": 24,
            "dep": 1,
            "dep_un": 1
        }
        self.head_names = list(self.output_channels.keys())
        self.no = sum(self.output_channels.values())
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.deform = deform
        self.common_head = common_head
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.patch_size = (kernel_size_1 - 1) + (kernel_size_2 - 1) + 1

        self.fgdm_pred = fgdm_predictor

        self.use_predecessors = use_predecessors
        self.detach_predecessors = detach_predecessors
        self.predecessors = {
            "cls": [],
            "o2d": [],
            "s2d": [],
            "o3d": ["cls"],
            "s3d": ["cls"],
            "hd": ["cls"],
            "dep": ["cls", "s3d"],
            "dep_un": ["cls", "s3d", "dep"]
        }
        self.dep_norm = 65.0

        ch = [ch[i] for i in range(self.nl)]
        self.cls_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["cls"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.o2d_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["o2d"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.s2d_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["s2d"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.o3d_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["o3d"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.s3d_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["s3d"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.hd_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["hd"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.dep_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["dep"]) if self.use_predecessors else ch_ for ch_ in ch]
        self.dep_un_in_ch = [ch_ + self.sum_predecessor_chs(self.predecessors["dep_un"]) if self.use_predecessors else ch_ for ch_ in ch]

        if self.common_head:
            self.common = nn.ModuleList(v10Detect3d.build_conv(ch_, ch_, 3, dsconv) for ch_ in ch)
            self.cls = self.build_small_head(ch, channels["cls_c"], self.nc)
            self.o2d = self.build_small_head(ch, channels["o2d_c"], 2)
            self.s2d = self.build_small_head(ch, channels["s2d_c"], 2)
            self.o3d = self.build_small_head(ch, channels["o3d_c"], 2)
            self.s3d = self.build_small_head(ch, channels["s3d_c"], 3)
            self.hd = self.build_small_head(ch, channels["hd_c"], 24)
            self.dep = self.build_small_head(ch, channels["dep_c"], 1)
            self.dep_un = self.build_small_head(ch, channels["dep_un_c"], 1)
        else:
            self.cls = self.build_head(self.cls_in_ch, channels["cls_c"], self.nc)
            self.o2d = self.build_head(self.o2d_in_ch, channels["o2d_c"], 2)
            self.s2d = self.build_head(self.s2d_in_ch, channels["s2d_c"], 2)
            self.o3d = self.build_head(self.o3d_in_ch, channels["o3d_c"], 2)
            self.s3d = self.build_head(self.s3d_in_ch, channels["s3d_c"], 3)
            self.hd = self.build_head(self.hd_in_ch, channels["hd_c"], 24)
            self.dep = self.build_head(self.dep_in_ch, channels["dep_c"], 1)
            self.dep_un = self.build_head(self.dep_un_in_ch, channels["dep_un_c"], 1)

        self.o2o_heads = nn.ModuleList(
            [self.cls, self.o2d, self.s2d, self.o3d, self.s3d, self.hd, self.dep, self.dep_un])
        self.o2m_heads = copy.deepcopy(self.o2o_heads)

        if self.fgdm_pred:
            self.fgdm_predictor = DepthPredictor(ch)

        self.keypoint_feature_extractor = KFE(ch)
        self.query_embedder = QueryEmbedder([channels[head_name + "_c"] if not self.half_channels else channels[head_name + "_c"] // 2 for head_name in self.head_names])
        self.kp_embedder = torch.nn.Embedding(8, 256)
        self.refiner = KeypointRefiner()

    def build_head(self, in_channels, mid_channels, output_channels):
        return nn.ModuleList(nn.Sequential(v10Detect3d.build_conv(x, mid_channels, self.kernel_size_1, self.dsconv,  deform=self.deform),
                                           v10Detect3d.build_conv(mid_channels, mid_channels // 2 if self.half_channels else mid_channels, self.kernel_size_2, self.dsconv),
                                           nn.Conv2d(mid_channels // 2 if self.half_channels else mid_channels, output_channels, 1)
                                           )
                             for x in in_channels
        )

    def build_small_head(self, in_channels, mid_channels, output_channels):
        return nn.ModuleList(nn.Sequential(v10Detect3d.build_conv(x, mid_channels, self.kernel_size_1, self.dsconv),
                                           nn.Conv2d(mid_channels, output_channels, 1)
                                           )
                             for x in in_channels
         )

    @staticmethod
    def build_conv(in_channels, out_channels, kernel_size, dsconv, deform=False):
        if dsconv:
            return nn.Sequential(Conv(in_channels, in_channels, kernel_size, g=in_channels, deform=deform), Conv(in_channels, out_channels, 1))
        else:
            return Conv(in_channels, out_channels, kernel_size,  deform=deform)

    def unravel_index(self, index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def extract_patches(self, x, indices):
        b, c, w, h = x.shape
        if not hasattr(self, "patch_size"):
            self.patch_size = 5
        pad = self.patch_size//2

        patches = []
        for i in range(b):
            img = x[i].unsqueeze(0)  # Shape [1, c, w, h]
            padded_img = torch.nn.functional.pad(img, (pad, pad, pad, pad))

            xs = indices[i, :, 0] + pad # offset the padding
            ys = indices[i, :, 1] + pad # offset the padding

            for xi, yi in zip(xs, ys):
                patch = padded_img[:, :, xi - pad:xi + 1 + pad, yi - pad:yi + 1 + pad]  # Extract 3x3 patch around (xi, yi)
                patches.append(patch)

        # Stack patches along the batch dimension [b*k, c, 3, 3]
        patches = torch.stack(patches).view(b * self.max_det, c, self.patch_size, self.patch_size)

        return patches

    def select_candidates(self, scores, batch_size):
        cls_scores_max = torch.max(scores, dim=1)[0]
        topk_indices = torch.zeros((batch_size, self.max_det, 2), dtype=torch.long)
        for b in range(batch_size):
            _, topk_ind = torch.topk(cls_scores_max[b].view(-1), self.max_det, dim=0, largest=True)
            topk_indices[b, :, 0], topk_indices[b, :, 1] = self.unravel_index(topk_ind, cls_scores_max[b].shape)
        return topk_indices

    def inference_forward_feat(self, x, heads):
        y = []
        head_features = []
        batch_sz = x[0].shape[0]
        head_names = list(self.output_channels.keys())
        for i in range(self.nl):
            outputs = {}
            head_feats = {}
            outputs[head_names[0]], head_feats[head_names[0]], _ = self.single_head_forward(heads[0][i], x[i])

            candidate_indices = self.select_candidates(outputs[head_names[0]], batch_sz)

            inputs = self.extract_patches(x[i], candidate_indices)
            for j, module in enumerate(heads[1:]):
                for layer in module[i]:
                    if isinstance(layer, Conv):
                        layer.conv.padding = 0
                out_, feats, _ = self.single_head_forward(module[i], inputs)

                output_shape = (x[i].shape[0], out_.shape[1], x[i].shape[2], x[i].shape[3])
                head_output = torch.zeros(output_shape, device=x[i].device)
                feat_output_shape = (x[i].shape[0], feats.shape[1], x[i].shape[2], x[i].shape[3])
                feat_output = torch.zeros(feat_output_shape, device=x[i].device)

                out = out_[:, :, 0, 0].view(output_shape[0], self.max_det, output_shape[1]).transpose(1, 2)
                feats = feats.view(output_shape[0], self.max_det, feats.shape[1]).transpose(1,2)

                for b in range(batch_sz):
                    head_output[b, :, candidate_indices[b, :, 0], candidate_indices[b, :, 1]] = out[b]
                    feat_output[b, :, candidate_indices[b, :, 0], candidate_indices[b, :, 1]] = feats[b]

                outputs[head_names[j+1]] = head_output
                head_feats[head_names[j+1]] = feat_output
            y.append(torch.cat(list(outputs.values()), dim=1))
            head_features.append(list(head_feats.values()))
        # Refinement
        refined_preds = self.refine_preds([yi.detach() for yi in y], [xi.detach() for xi in x], head_features)
        return y, refined_preds

    def forward_feat(self, x, heads):
        y = []
        depth_features = []
        head_features = []
        head_names = list(self.output_channels.keys())
        for i in range(self.nl):
            outputs = {}
            ems = {}
            for j, module in enumerate(heads):
                outputs[head_names[j]], ems[head_names[j]], dep_embs = self.single_head_forward(module[i], x[i])
                if head_names[j] == "dep":
                    depth_features.append(dep_embs)

            y.append(torch.cat(list(outputs.values()), dim=1))
            head_features.append(list(ems.values()))

        # Refinement
        refined_preds = self.refine_preds([yi.detach() for yi in y], [xi.detach() for xi in x], head_features)
        return y, refined_preds, depth_features

    def single_head_forward(self, head, features):
        assert len(head) == 3
        o1 = head[0](features)
        embeddings = head[1](o1)
        return head[2](embeddings), embeddings.detach(), o1


    def sum_predecessor_chs(self, predecessors):
        return sum([self.output_channels[predecessor] for predecessor in predecessors]) if len(predecessors) > 0 else 0

    def decode(self, cls, pred_o2d, pred_s2d, pred_o3d, pred_s3d, pred_hd, pred_dep, pred_dep_un):
        s2d = pred_s2d * self.strides
        o2d = (pred_o2d + self.anchors) * self.strides
        xy1 = o2d - s2d / 2
        xy2 = o2d + s2d / 2
        bbox = torch.cat((xy1, xy2), dim=1)

        center3d = (pred_o3d + self.anchors) * self.strides

        return torch.cat((cls, bbox, center3d, pred_s3d, pred_hd, pred_dep, pred_dep_un), dim=1)

    def inference(self, x):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in (
        "saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            cls, pred_o2d, pred_s2d, pred_o3d, pred_s3d, pred_hd, pred_dep, pred_dep_un = (
                torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2).split(
                    (self.nc, 2, 2, 2, 3, 24, 1, 1), 1
                ))
            preds = self.decode(cls, pred_o2d, pred_s2d, pred_o3d, pred_s3d, pred_hd, pred_dep, pred_dep_un)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(box * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        # else:
        #
        #     dbox = self.decode_bboxes(box, self.anchors.unsqueeze(0)) * self.strides

        y = preds
        return y if self.export else (y, x)

    def forward(self, x):
        if self.training:
            one2one,  refined_o2o, o2o_embs = self.forward_feat([xi.detach() for xi in x], self.o2o_heads)
            one2many, refined_o2m, o2m_embs = self.forward_feat(x, self.o2m_heads)

            if hasattr(self, "fgdm_pred") and self.fgdm_pred:
                depth_maps = self.fgdm_predictor(x, return_embeddings=True)
            else:
                depth_maps = torch.empty(1)

            return {"one2many": one2many, "refined_o2m": refined_o2m, "o2m_embs": o2m_embs,
                    "one2one": one2one, "refined_o2o": refined_o2o, "o2o_embs": o2o_embs,
                    "depth_maps": depth_maps}
        else:
            (one2one, refined_o2o), o2o_embs = self.inference_forward_feat([xi.detach() for xi in x], self.o2o_heads), None
            # self.get_head_ranks()
            one2one = self.inference(one2one)
            refined_o2o = self.inference(refined_o2o)
            if not self.export:
                return {"one2one": one2one, "refined_o2o": refined_o2o, "o2o_embs": o2o_embs}
            else:
                raise NotImplementedError("TODO")
                assert (self.max_det != -1)
                boxes, scores, labels = ops.v10postprocess(one2one.permute(0, 2, 1), self.max_det, self.nc)
                return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)


    def refine_preds(self, output, backbone_features, head_features):
        cls, pred_o2d, pred_s2d, pred_o3d, pred_s3d, pred_hd, pred_dep, pred_dep_un = (
            torch.cat([xi.view(output[0].shape[0], self.no, -1) for xi in output], 2).split(
                (self.nc, 2, 2, 2, 3, 24, 1, 1), 1
            ))
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(output, self.stride, 0.5))
        preds = self.decode(cls, pred_o2d, pred_s2d, pred_o3d, pred_s3d, pred_hd, pred_dep, pred_dep_un)

        kp_feats = self.extract_kp_feats(preds, backbone_features)
        queries = self.query_embedder(head_features)
        embeddings = self.get_kp_embeddings(kp_feats)

        refined_output = self.refiner(embeddings, kp_feats, queries)

        refined = []
        for out, ref in zip(output, refined_output):
            refined.append(torch.cat((out[:, :9], ref), dim=1))

        return refined

    def get_kp_embeddings(self, kp_feats):
        embeddings = []
        for scale in kp_feats:
            embedding = self.kp_embedder(torch.arange(8, device=scale.device).unsqueeze(0))
            embeddings.append(embedding.transpose(1, 2).unsqueeze(2).unsqueeze(2).repeat(scale.shape[0], 1, scale.shape[2], scale.shape[3], 1))
        return embeddings

    def extract_kp_feats(self, preds, backbone_features):
        #FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        calib = torch.tensor(np.array([ 6.5179e+02,  1.7700e+02,  7.4361e+02,  7.3885e+02,  6.4071e-02, -3.0708e-04]), dtype=torch.float64)
        calibs = calib.unsqueeze(0).repeat(backbone_features[0].shape[0], 1).to(backbone_features[0].device)
        mean_sizes = torch.tensor(np.array([
            [1.52563191462, 1.62856739989, 3.88311640418],
            [1.76255119, 0.66068622, 0.84422524],
            [1.73698127, 0.59706367, 1.76282397]]))
        #FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        return self.keypoint_feature_extractor.forward(backbone_features, preds, calibs, mean_sizes,
                                                           stride_tensor=self.strides,
                                                           strides=self.stride)


    def decode_bboxes(self, bboxes, anchors):
        # anchor_points:
        # pred_2d: offset_2d (2), size_2d(2)
        offset, size = bboxes.split((2, 2), dim=1)
        centers = anchors + offset
        xy1 = centers - size / 2
        xy2 = centers + size / 2
        xy = xy1 + (xy2 - xy1) / 2
        wh = xy2 - xy1
        return torch.cat((xy, wh), dim=1)

    def bias_init(self):
        if self.nl == 1:
            deps = [40]
            ranges = [[-3.5, 3.5]]
        elif self.nl == 2:
            deps = [45, 20]
            ranges = [[-2, 2], [-2, 2]]
        elif self.nl == 3:
            deps = [45, 25, 10]
            ranges = [[-2, 2], [-1.5, 1.5], [-1, 1]]
        else:
            raise RuntimeError("Initialization only set for 1 and 3 scales")
        for i in range(self.nl):
            self.cls[i][-1].bias.data[: self.nc] = math.log(5 / self.nc / ((1280 / self.stride[i]) * (384 / self.stride[i])))
            self.s2d[i][-1].bias.data.fill_(6)
            self.o2d[i][-1].bias.data.fill_(0)
            self.o3d[i][-1].bias.data.fill_(0)
            self.s3d[i][-1].bias.data.fill_(0.0)
            nn.init.normal_(self.s3d[i][-1].weight, std=0.05)
            self.dep[i][-1].bias.data.fill_(deps[i])
            nn.init.uniform_(self.dep[i][-1].weight, a=ranges[i][0], b=ranges[i][1])

        self.o2o_heads = nn.ModuleList([self.cls, self.o2d, self.s2d, self.o3d, self.s3d, self.hd, self.dep, self.dep_un])
        self.o2m_heads = copy.deepcopy(self.o2o_heads)
        pass

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def calc_rank(self, layer):
        weight_tensor = layer[0].conv.weight
        output_channels, input_channels, k1, k2 = weight_tensor.shape
        wght_matrix = np.transpose(weight_tensor.cpu().numpy(), (1, 2, 3, 0))
        wght_matrix = wght_matrix.reshape(output_channels,
                                          input_channels * k1 * k2)
        U, S, V_t = np.linalg.svd(wght_matrix, full_matrices=True)
        rank = np.linalg.matrix_rank(wght_matrix, tol=S[0] / 2) / input_channels
        svalues = S / S[0]
        return rank, svalues

    def calc_weight_distri(self, layer):
        weight_tensor = layer[0].conv.weight
        center = weight_tensor[:, :, 1, 1].abs().mean()
        others = weight_tensor[:, :, [0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 0, 2, 0, 1, 2]].abs().mean()
        return center, others

    def get_head_ranks(self):
        head_names = list(self.output_channels.keys())
        scales = [8, 16, 32]
        ranks = np.zeros((2, 3, 8), dtype=np.float32)
        svalues = np.zeros((2, 3, 8, 128))
        weight_distri = np.zeros((2, 3, 8, 2)) # center, others
        for i, head in enumerate(self.o2o_heads):
            for j, scale in enumerate(head):
                ranks[0, j, i],  svalues[0, j, i, :] = self.calc_rank(scale)
                weight_distri[0, j, i, 0], weight_distri[0, j, i, 1] = self.calc_weight_distri(scale)
        for i, head in enumerate(self.o2m_heads):
            for j, scale in enumerate(head):
                ranks[1, j, i], svalues[1, j, i, :] = self.calc_rank(scale)
                weight_distri[1, j, i, 0], weight_distri[1, j, i, 1] = self.calc_weight_distri(scale)

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('dark_background')

        def get_cmap(n, name='hsv'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n)

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        cmap = get_cmap(8, "Pastel1")
        colors = [cmap(i) for i in range(8)]
        rows = [f"Subsampling {scale}" for scale in scales]
        cols = ["o2o", "o2m"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        for mapping in range(2):
            for scale in range(3):
                axes[scale, mapping].bar(np.arange(8), ranks[mapping, scale, :].reshape(8), tick_label=head_names, color=colors)
        fig.tight_layout()
        plt.show()

        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        cmap = get_cmap(8, "jet")
        colors = [cmap(i) for i in range(8)]
        rows = [f"Subsampling {scale}" for scale in scales]
        cols = ["o2o", "o2m"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        for mapping in range(2):
            for scale in range(3):
                for head in range(8):
                    axes[scale, mapping].set_ylim((0, 1))
                    axes[scale, mapping].plot(np.arange(128)/128, svalues[mapping, scale, head],
                                              label=head_names[head], color=colors[head], linewidth=2)
        plt.legend()
        fig.tight_layout()
        plt.show()

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        cmap = get_cmap(8, "Pastel1")
        colors = [x for xs in [[cmap(i), cmap(i)] for i in range(8)] for x in xs]
        rows = [f"Subsampling {scale}" for scale in scales]
        cols = ["o2o", "o2m"]
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, size='large')

        labels = head_names * 2
        for mapping in range(2):
            for scale in range(3):
                axes[scale, mapping].bar(np.arange(16), weight_distri[mapping, scale].reshape(16), tick_label=labels,
                                         color=colors)
        plt.legend()
        fig.tight_layout()
        plt.show()
        print()


class DepthPredictor(nn.Module):
    '''
    Adapted from MonoDETR depth predictor
    '''
    def __init__(self, ch=()):
        super().__init__()
        self.depth_min = 1   #self.args.min_depth_threshold #FIXME
        self.depth_max = 70 #self.args.max_depth_threshold #FIXME
        self.depth_bins = 80

        bin_size = 2 * (self.depth_max - self.depth_min) / (self.depth_bins * (1 + self.depth_bins))
        bin_indice = torch.linspace(0, self.depth_bins - 1, self.depth_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + self.depth_min
        bin_value = torch.cat([bin_value, torch.tensor([self.depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        # Create modules
        hidden_dim = 128
        self.downsample = nn.Sequential(
            nn.Conv2d(ch[0], hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, hidden_dim))
        self.proj = nn.Sequential(
            nn.Conv2d(ch[1], hidden_dim, kernel_size=(1, 1)),
            nn.GroupNorm(32, hidden_dim))
        self.upsample = nn.Sequential(
            nn.Conv2d(ch[2], hidden_dim, kernel_size=(1, 1)),
            nn.GroupNorm(32, hidden_dim))

        self.depth_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=hidden_dim),
            nn.ReLU())

        self.depth_classifier = nn.Conv2d(hidden_dim, self.depth_bins + 1, kernel_size=(1, 1))

    def forward(self, feature, return_embeddings=False):
        assert len(feature) == 3

        src_8 = self.downsample(feature[0])
        src_16 = self.proj(feature[1])
        src_32 = self.upsample(torch.nn.functional.interpolate(feature[2], size=src_16.shape[-2:], mode='bilinear'))
        # new_add
        # src_8 = self.proj(feature[0])
        # src_16 = self.upsample(F.interpolate(feature[1], size=src_8.shape[-2:], mode='bilinear'))
        # src_32 = self.upsample(F.interpolate(feature[2], size=src_8.shape[-2:], mode='bilinear'))
        ####
        src = (src_8 + src_16 + src_32) / 3

        for i, layer in enumerate(self.depth_head):
            src = layer(src)
            if i == 2 and return_embeddings:
                features = src
        # ipdb.set_trace()
        depth_logits = self.depth_classifier(src)

        depth_probs = torch.nn.functional.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        if return_embeddings:
            return depth_logits, weighted_depth, features
        else:
            return depth_logits, weighted_depth

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
