# ------------------------------------------------------------
# sparse_conv2d_layer.py   (updated)
# ------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sparse_conv2d as _spc   # the compiled CUDA extension from the previous steps
from torch.profiler import record_function


def _pair(v):
    """Utility that mimics torch.nn.modules.utils._pair."""
    if isinstance(v, (list, tuple)):
        return v
    return (v, v)


class SparseConv2d(nn.Module):
    """
    Drop‑in replacement for ``torch.nn.Conv2d`` that evaluates only a
    user‑supplied set of output locations (``indices``) and supports
    grouped/depth‑wise convolutions.

    If ``indices`` is ``None`` (or an empty tensor), the layer simply
    delegates to the standard ``torch.nn.functional.conv2d`` implementation,
    preserving the exact behaviour of a regular Conv2d layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()

        # Save hyper‑parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.bias_flag = bias

        # ----- Parameter allocation -----
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
                dtype=torch.float32,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Weight initialisation (mirrors torch.nn.Conv2d)
    # ------------------------------------------------------------------
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = (
                self.in_channels
                * self.kernel_size[0]
                * self.kernel_size[1]
                // self.groups
            )
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Forward pass – now with a fallback to regular conv2d
    # ------------------------------------------------------------------
    def forward(self, input: torch.Tensor, indices: torch.Tensor = torch.empty(0)) -> torch.Tensor:
        """
        Parameters
        ----------
        input   : Tensor[B, C_in, H_in, W_in] (float32/float64, on CUDA)
        indices : Tensor[B, N, 2] (int64) – (y, x) coordinates **in the output**
                  space.  If ``None`` (or an empty tensor) the layer behaves
                  exactly like ``torch.nn.functional.conv2d``.

        Returns
        -------
        Tensor[B, C_out, H_out, W_out] – dense output.
        """
        # ------------------------------------------------------------------
        # 1️⃣  Fast path – regular dense convolution when no indices are given
        # ------------------------------------------------------------------
        if indices is None or indices.numel() == 0 or not input.is_cuda:
            # Use the built‑in functional implementation – this guarantees
            # identical numerical results to a plain nn.Conv2d.
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        # ------------------------------------------------------------------
        # 2️⃣  Sparse path – call the custom CUDA kernel
        # ------------------------------------------------------------------
        if not input.is_cuda:
            raise RuntimeError("SparseConv2d currently only supports CUDA tensors.")
        if not indices.is_cuda:
            raise RuntimeError("Indices tensor must be on the same CUDA device as input.")

        #with record_function("my_custom_kernel"):
        out = _spc.forward(
            input,
            self.weight,
            self.bias
            if self.bias is not None
            else torch.tensor([], device=input.device, dtype=input.dtype),
            indices,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )

        return out

    # ------------------------------------------------------------------
    # Helper to compute the expected output shape (useful for validation)
    # ------------------------------------------------------------------
    def _output_shape(self, input_shape):
        B, _, H_in, W_in = input_shape
        H_out = (
            (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
            // self.stride[0]
            + 1
        )
        W_out = (
            (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
            // self.stride[1]
            + 1
        )
        return (B, self.out_channels, H_out, W_out)

    # ------------------------------------------------------------------
    # Pretty‑print representation (keeps the same look as nn.Conv2d)
    # ------------------------------------------------------------------
    def extra_repr(self):
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}"
            f", stride={self.stride}, padding={self.padding}, dilation={self.dilation}"
            f", groups={self.groups}, bias={self.bias is not None}"
        )
    
if __name__=='__main__':
    import time
    from torch.profiler import profile, ProfilerActivity, record_function

    device = torch.device('cuda')
    model = SparseConv2d(64, 16, kernel_size=3, padding=1).to(device)

    B, C, H, W = 64, 64, 128, 128
    x = torch.randn(B, C, H, W, device=device)

    # ---- dense path (fallback) ----
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True) as prof:
        y_dense = model(x)                # indices=None → regular conv2d
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    print('Dense forward time:', time.time() - t0)

    # ---- sparse path ----
    H_out = (H + 2*1 - (3-1) - 1)//1 + 1
    W_out = (W + 2*1 - (3-1) - 1)//1 + 1
    N = 50
    indices = torch.stack([
        torch.randint(0, H_out, (B, N), device=device),
        torch.randint(0, W_out, (B, N), device=device)
    ], dim=-1)

    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True) as prof:
        y_sparse = model(x, indices)      # uses custom kernel
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    print('Sparse forward time:', time.time() - t0)