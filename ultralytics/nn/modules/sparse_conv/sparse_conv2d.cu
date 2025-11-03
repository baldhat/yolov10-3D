// sparse_conv2d_fast.cu
#include <torch/extension.h>
#include <cuda.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sparse_conv2d_fast_kernel(
    const scalar_t* __restrict__ input,      // [B, C_in, H_in, W_in]
    const scalar_t* __restrict__ weight,     // [C_out, C_in/groups, K_h, K_w]
    const scalar_t* __restrict__ bias,       // [C_out] (optional)
    const int64_t* __restrict__ indices,     // [B, N, 2] (y, x)
    scalar_t* __restrict__ output,           // [B, C_out, H_out, W_out]
    const int B, const int C_in,
    const int H_in, const int W_in,
    const int C_out, const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int H_out, const int W_out,
    const int N,
    const int groups
) {
    // Thread idx maps to (b, n, co)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * C_out) return;

    int b = idx / (N * C_out);
    int n = (idx / C_out) % N;
    int co = idx % C_out;

    int g = co / (C_out / groups);                // group of this output channel
    int Cin_per_grp = C_in / groups;
    int Cout_per_grp = C_out / groups;

    int ci_start = g * Cin_per_grp;
    int ci_end   = ci_start + Cin_per_grp;

    // Load output coordinates
    int64_t out_y = indices[b * N * 2 + n * 2 + 0];
    int64_t out_x = indices[b * N * 2 + n * 2 + 1];
    if (out_y < 0 || out_y >= H_out || out_x < 0 || out_x >= W_out) return;

    int in_y_origin = out_y * stride_h - pad_h;
    int in_x_origin = out_x * stride_w - pad_w;

    // Accumulator
    scalar_t acc = bias ? bias[co] : static_cast<scalar_t>(0);

    // Loop over input channels in this group
    for (int ci = ci_start; ci < ci_end; ++ci) {
        const scalar_t* input_ptr = input + b * C_in * H_in * W_in + ci * H_in * W_in;
        const scalar_t* weight_ptr = weight + co * Cin_per_grp * K_h * K_w + (ci - ci_start) * K_h * K_w;

        // Loop over kernel spatial
        #pragma unroll
        for (int kh = 0; kh < K_h; ++kh) {
            int in_y = in_y_origin + kh * dilation_h;
            if (in_y < 0 || in_y >= H_in) continue;

            #pragma unroll
            for (int kw = 0; kw < K_w; ++kw) {
                int in_x = in_x_origin + kw * dilation_w;
                if (in_x < 0 || in_x >= W_in) continue;

                acc += input_ptr[in_y * W_in + in_x] * weight_ptr[kh * K_w + kw];
            }
        }
    }

    // Write to output
    output[b * C_out * H_out * W_out + co * H_out * W_out + out_y * W_out + out_x] = acc;
}

// ------------------------------------------------------------------
// Python binding
// ------------------------------------------------------------------
torch::Tensor sparse_conv2d_fast_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor indices,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups
) {
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(0);
    const int K_h = weight.size(2);
    const int K_w = weight.size(3);

    const int H_out = (H_in + 2 * pad_h - dilation_h * (K_h - 1) - 1) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - dilation_w * (K_w - 1) - 1) / stride_w + 1;

    const int N = indices.size(1);

    auto options = input.options();
    auto output = torch::empty({B, C_out, H_out, W_out}, options);

    const int threads = 256;  // smaller threads to increase occupancy
    const int blocks = (B * N * C_out + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "sparse_conv2d_fast_kernel", ([&] {
        auto stream = at::cuda::getCurrentCUDAStream();
        sparse_conv2d_fast_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), 
            bias.numel() ? bias.data_ptr<scalar_t>() : nullptr,
            indices.data_ptr<int64_t>(), output.data_ptr<scalar_t>(),
            B, C_in, H_in, W_in, C_out, K_h, K_w,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            H_out, W_out, N, groups
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sparse_conv2d_fast_forward, "Sparse Conv2d fast forward (CUDA) with groups");
}