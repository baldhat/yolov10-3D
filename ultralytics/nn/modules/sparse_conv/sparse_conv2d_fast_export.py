import torch

# Get the operator overload (default overload)
op = torch.ops.sparseconv.sparse_conv2d_fast.default

# Register the operator for torch.export as an opaque op
torch.export.register_operator(op)