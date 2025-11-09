#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>  // 👈 needed for device_ptr (older CUDA)
#include <thrust/functional.h>

using at::Tensor;

// Kernel to convert flat indices to (row, col)
__global__ void flat_to_rowcol_kernel(
    const int64_t* __restrict__ flat_idx,
    int64_t* __restrict__ row_idx,
    int64_t* __restrict__ col_idx,
    int64_t W,
    int64_t k) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    int64_t idx = flat_idx[i];
    row_idx[i] = idx / W;
    col_idx[i] = idx % W;
  }
}

Tensor topk_indices_cuda_forward(
    Tensor logits,
    int64_t k) {

  // Flatten spatial dims
  auto logits_flat = logits.view({-1});
  int64_t N = logits_flat.size(0);
  auto logits_flat_contig = logits_flat.contiguous();

  // Create index tensor [0, 1, 2, ..., N-1]
  auto indices = at::arange(N, logits.options().dtype(at::kLong));

  float* val_ptr = logits_flat_contig.data_ptr<float>();
  int64_t* idx_ptr = indices.data_ptr<int64_t>();

  // ✅ Use Thrust directly with raw pointers
  thrust::sort_by_key(
      thrust::device,                     // execution policy
      val_ptr, val_ptr + N,               // values
      idx_ptr,                            // indices
      thrust::greater<float>());          // descending order

  // Take top-k indices and values
  Tensor topk_flat_idx = indices.slice(0, 0, k).contiguous();

  // Allocate row and col tensors
  auto opts = logits_flat.options().dtype(at::kLong);
  Tensor rows = at::empty({k}, opts);
  Tensor cols = at::empty({k}, opts);

  // Compute (row, col)
  const int threads = 256;
  const int blocks = (k + threads - 1) / threads;
  flat_to_rowcol_kernel<<<blocks, threads>>>(
      topk_flat_idx.data_ptr<int64_t>(),
      rows.data_ptr<int64_t>(),
      cols.data_ptr<int64_t>(),
      logits.size(2),  // W
      k);
  cudaDeviceSynchronize();

  Tensor coords = at::stack({rows, cols}, 1);  // (k, 2)

  return coords;
}
