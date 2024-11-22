
import torch
import torch.nn as nn
from torch.nn.functional import silu
from ultralytics.nn.modules import Conv

class KeypointRefiner(nn.Module):
    def __init__(self):
        super(KeypointRefiner, self).__init__()
        self.attention_layer = nn.MultiheadAttention(256, 8, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.pre_norm1 = nn.LayerNorm(256)
        self.pre_norm2 = nn.LayerNorm(256)
        self.pre_norm3 = nn.LayerNorm(256)
        self.lin1 = nn.Linear(256, 512)
        self.lin2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([
            nn.Sequential( nn.Conv2d(256, 64, kernel_size=1), nn.SiLU(), nn.Conv2d(64, 3, kernel_size=1)),   # size 3d
            nn.Sequential( nn.Conv2d(256, 64, kernel_size=1), nn.SiLU(), nn.Conv2d(64, 24, kernel_size=1)),  # rotation
            nn.Sequential( nn.Conv2d(256, 64, kernel_size=1), nn.SiLU(), nn.Conv2d(64, 1, kernel_size=1)),   # depth
            nn.Sequential( nn.Conv2d(256, 64, kernel_size=1), nn.SiLU(), nn.Conv2d(64, 1, kernel_size=1)),   # depth uncertainty
        ])
        self.heads[0][-1].bias.data.fill_(0.0)
        nn.init.normal_(self.heads[0][-1].weight, std=0.05)
        self.heads[2][-1].bias.data.fill_(35)
        nn.init.uniform_(self.heads[2][-1].weight, a=-3.5, b=3.5)

    def forward(self, embeddings, keypoints, queries):
        output = []
        for embedding, keypoint, query in zip(embeddings, keypoints, queries):
            neck_out = self.forward_neck(embedding, keypoint, query)
            output.append(self.forward_heads(neck_out))
        return output

    def forward_neck(self, embedding, keypoint, query):
        q = self.pre_norm1(query.unsqueeze(-1).reshape(-1, 1, query.shape[1]))
        inp = keypoint.reshape(-1, embedding.shape[4], embedding.shape[1])
        emb = embedding.reshape(-1, embedding.shape[4], embedding.shape[1])
        k = self.pre_norm2(inp + emb)
        v = self.pre_norm3(inp)
        attn_output, attn_weights = self.attention_layer(q, k, v)
        x1 = self.norm1(attn_output + q)
        x2 = silu(self.lin2(self.dropout(silu(self.lin1(x1))))) + x1
        return self.norm2(x2).reshape(query.shape)

    def forward_heads(self, x):
        output = []
        for head in self.heads:
            out = head(x)
            output.append(out)
        return torch.cat(output, dim=1)
