// laguerre_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for fused normalization and Laguerre expansion
template <typename scalar_t>
__global__ void laguerre_fused_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int max_seq_len,
    const int d_model,
    const int total_elements // batch_size * seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Normalize input: 2 * (x / (max_seq_len - 1)) - 1
    // Note: Laguerre is naturally defined for x >= 0, but we map to [-1, 1] for consistency
    scalar_t x = 2.0 * (input[idx] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;

    // Initialize Laguerre terms
    // L_0(x) = 1
    output[idx * d_model + 0] = 1.0;
    if (d_model > 1) {
        // L_1(x) = 1 - x
        output[idx * d_model + 1] = 1.0 - x;
    }
    // Recurrence: (n+1) * L_{n+1}(x) = (2n+1-x) * L_n(x) - n * L_{n-1}(x)
    for (int n = 1; n < d_model - 1; ++n) {
        output[idx * d_model + (n + 1)] = ((2.0 * n + 1.0 - x) * output[idx * d_model + n] - n * output[idx * d_model + (n - 1)]) / (n + 1.0);
    }
}

std::vector<torch::Tensor> laguerre_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto total_elements = batch_size * seq_len;

    auto output = torch::zeros({batch_size, seq_len, d_model}, input.options());

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "laguerre_fused_forward_cuda", ([&] {
        laguerre_fused_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            max_seq_len,
            d_model,
            total_elements
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed (Laguerre): %s\n", cudaGetErrorString(err));
        TORCH_CHECK(false, "CUDA kernel launch failed (Laguerre)");
    }

    return {output};
} 