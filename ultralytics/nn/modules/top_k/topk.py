
import torch
import topk_cuda as tc

def topk_indices(logits: torch.Tensor, k: int):
    # logits: (1, H, W)
    H, W = logits.shape[1:]
    flat = logits.reshape(-1)  # just view, no copy
    _, idx = torch.topk(flat, k, largest=True, sorted=False)
    rows = torch.div(idx, W, rounding_mode='trunc')
    cols = idx - rows * W
    return torch.stack((rows, cols), dim=1)


def topk_cuda(scores, topk):
    if not scores.is_cuda:
        return topk_indices(scores, topk)
    return tc.forward(scores, topk)