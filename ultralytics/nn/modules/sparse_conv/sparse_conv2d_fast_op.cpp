// sparse_conv2d_fast_op.cpp
#include <torch/extension.h>

// Forward declaration of your CUDA kernel wrapper
torch::Tensor sparse_conv2d_fast_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor indices,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t groups);

// -----------------------------------------------------------------------------
// Dispatcher registration (opaque custom operator)
// -----------------------------------------------------------------------------

TORCH_LIBRARY(sparseconv, m) {
    m.def("sparse_conv2d_fast(Tensor input, Tensor weight, Tensor bias, Tensor indices, "
          "int stride_h, int stride_w, int pad_h, int pad_w, "
          "int dilation_h, int dilation_w, int groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparseconv, CUDA, m) {
    m.impl("sparse_conv2d_fast", sparse_conv2d_fast_forward);
}