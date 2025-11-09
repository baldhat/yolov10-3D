#include <torch/extension.h>
#include <vector>

// Declare the CUDA forward function
torch::Tensor topk_indices_cuda_forward(
    torch::Tensor logits,
    int64_t k);

// C++ interface
torch::Tensor topk_indices_forward(
    torch::Tensor logits,
    int64_t k) {
  TORCH_CHECK(logits.device().is_cuda(), "logits must be a CUDA tensor");
  TORCH_CHECK(logits.dim() == 3 && logits.size(0) == 1,
              "Input must have shape (1, H, W)");
  return topk_indices_cuda_forward(logits, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topk_indices_forward, "topk_indices forward (CUDA)");
}
