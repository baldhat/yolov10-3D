#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

// Host helper: unravel a flattened index into (h, w)
static inline void unravel_index_2d_host(long idx, int H, int W, long &out_h, long &out_w) {
    out_h = idx / W;
    out_w = idx % W;
}

// Kernel: compute cls_scores_max[b, h, w] = max_c scores[b, c, h, w]
__global__ void compute_cls_scores_max_kernel(
    const float* scores, float* cls_scores_max,
    int B, int C, int H, int W
) {
    int b = blockIdx.x;
    if (b >= B) return;

    int hw = H * W;
    for (int idx = threadIdx.x; idx < hw; idx += blockDim.x) {
        int h = idx / W;
        int w = idx % W;

        float max_val = -1e30f;
        for (int c = 0; c < C; ++c) {
            float val = scores[((b * C + c) * H + h) * W + w];
            if (val > max_val) max_val = val;
        }
        cls_scores_max[b * H * W + h * W + w] = max_val;
    }
}

// Host function: run reduction + topk sorting
torch::Tensor select_candidates_cuda(torch::Tensor scores, int max_det) {
    TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
    TORCH_CHECK(scores.dim() == 4, "scores must be [B, C, H, W]");

    const int B = scores.size(0);
    const int C = scores.size(1);
    const int H = scores.size(2);
    const int W = scores.size(3);
    const int HW = H * W;

    // Allocate tensors
    auto cls_scores_max = torch::empty({B, H, W}, scores.options());
    auto topk_indices = torch::zeros({B, max_det, 2},
                                     torch::dtype(torch::kLong).device(scores.device()));

    // 1️⃣ Compute per-pixel max across classes
    const int threads = 256;
    const int blocks = B;
    compute_cls_scores_max_kernel<<<blocks, threads>>>(
        scores.data_ptr<float>(),
        cls_scores_max.data_ptr<float>(),
        B, C, H, W
    );

    // 2️⃣ For each batch, run thrust::sort_by_key for top-K
    for (int b = 0; b < B; ++b) {
        float* batch_scores = cls_scores_max.data_ptr<float>() + b * HW;

        // Create thrust device pointer to this batch's scores
        thrust::device_ptr<float> score_ptr(batch_scores);
        thrust::device_vector<long> indices(HW);
        thrust::sequence(indices.begin(), indices.end());

        thrust::sort_by_key(
            thrust::device, score_ptr, score_ptr + HW,
            indices.begin(), thrust::greater<float>()
        );

        // Copy top-K (h, w)
        auto topk_ptr = topk_indices.data_ptr<long>() + b * max_det * 2;
        for (int k = 0; k < max_det && k < HW; ++k) {
            long h, w;
            unravel_index_2d_host(indices[k], H, W, h, w);
            topk_ptr[k * 2 + 0] = h;
            topk_ptr[k * 2 + 1] = w;
        }
    }

    return topk_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("select_candidates_cuda", &select_candidates_cuda,
          "Select top-K (h, w) candidates (CUDA, 4D input)");
}
