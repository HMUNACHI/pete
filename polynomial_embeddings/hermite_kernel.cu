// hermite_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for fused normalization and Hermite expansion (Physicists')
template <typename scalar_t>
__global__ void hermite_fused_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int max_seq_len,
    const int d_model,
    const int total_elements // batch_size * seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Normalize input: 2 * (x / (max_seq_len - 1)) - 1
    scalar_t x = 2.0 * (input[idx] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;

    // Initialize Hermite terms
    // H_0(x) = 1
    output[idx * d_model + 0] = 1.0;
    if (d_model > 1) {
        // H_1(x) = 2x
        output[idx * d_model + 1] = 2.0 * x;
    }
    // Recurrence: H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
    for (int n = 1; n < d_model - 1; ++n) {
        output[idx * d_model + (n + 1)] = 2.0 * x * output[idx * d_model + n] - 2.0 * n * output[idx * d_model + (n - 1)];
    }
}

std::vector<torch::Tensor> hermite_fused_forward_cuda(
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hermite_fused_forward_cuda", ([&] {
        hermite_fused_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            max_seq_len,
            d_model,
            total_elements
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed (Hermite): %s\n", cudaGetErrorString(err));
        TORCH_CHECK(false, "CUDA kernel launch failed (Hermite)");
    }

    return {output};
} 