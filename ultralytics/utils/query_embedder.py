import torch
import torch.nn as nn


class QueryEmbedder(nn.Module):
    def __init__(self, in_channels):
        super(QueryEmbedder, self).__init__()

        self.linears = nn.ModuleList(
            nn.Conv2d(chs, 64, 1, 1, (0,0)) for chs in in_channels
        )
        self.joint_part = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(len(in_channels) * 64, 256, 1, 1,(0,0)),
            nn.SiLU(),
            nn.BatchNorm2d(256)
        )

    def forward(self, x):
        scales = []
        for scale in x:
            outs = []
            for i, xi in enumerate(scale):
                outs.append(self.linears[i](xi))
            concatenated = torch.cat(outs, 1)
            scales.append(self.joint_part(concatenated))
        return scales