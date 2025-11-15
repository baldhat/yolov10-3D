from torch.onnx import register_custom_op_symbolic
import torch

def symbolic_sparse_conv2d_fast(g, input, weight, bias, indices,
                                stride_h, stride_w,
                                pad_h, pad_w,
                                dilation_h, dilation_w,
                                groups):

    return g.op(
        "sparseconv::sparse_conv2d_fast",
        input,
        weight,
        bias,
        indices,
        stride_h_i=stride_h,
        stride_w_i=stride_w,
        pad_h_i=pad_h,
        pad_w_i=pad_w,
        dilation_h_i=dilation_h,
        dilation_w_i=dilation_w,
        groups_i=groups
    )

# Register for ONNX opset 18+
register_custom_op_symbolic(
    "sparseconv::sparse_conv2d_fast",
    symbolic_sparse_conv2d_fast,
    18
)