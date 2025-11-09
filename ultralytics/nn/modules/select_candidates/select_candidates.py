import torch
import select_candidates_cuda
import ultralytics.nn.modules.top_k

def topk_indices(logits: torch.Tensor, k: int = 50):
    """
    Returns the indices of the top-k values in a (1, H, W) logit map.

    Args:
        logits (torch.Tensor): Input tensor of shape (1, H, W).
        k (int): Number of top values to extract.

    Returns:
        torch.Tensor: Tensor of shape (k, 2) with (row, col) indices of top-k values.
        torch.Tensor: Tensor of shape (k,) with the corresponding top-k values.
    """
    W = logits.shape[2]

    # Flatten spatial dimensions
    flat = logits.view(-1)  # shape (H*W,)
    
    # Get top-k values and flat indices
    _, topk_flat_idx = torch.topk(flat, k)

    # Convert flat indices back to 2D coordinates
    rows = topk_flat_idx // W
    cols = topk_flat_idx % W
    indices = torch.stack((rows, cols), dim=1)  # shape (k, 2)

    return indices

def select_candidates_kernel(scores, topk):
    cls_scores_max = torch.max(scores, dim=1)[0]
    return topk_indices(cls_scores_max, topk)