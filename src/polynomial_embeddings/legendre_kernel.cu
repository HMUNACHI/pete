// legendre_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for fused normalization and Legendre expansion
template <typename scalar_t>
__global__ void legendre_fused_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int max_seq_len,
    const int d_model,
    const int total_elements 
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Normalize input to (-1, 1): 2 * (x / (max_seq_len - 1)) - 1
    scalar_t x = 2.0 * (input[idx] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;

    // Initialize Legendre terms
    // Each thread handles one [batch, seq] position
    output[idx * d_model + 0] = 1.0; // P0(x) = 1
    if (d_model > 1) {
        output[idx * d_model + 1] = x; // P1(x) = x
    }
    for (int n = 1; n < d_model - 1; ++n) {
        // Legendre recurrence:
        // P_{n+1}(x) = ((2n + 1) * x * P_n(x) - n * P_{n-1}(x)) / (n + 1)
        scalar_t Pn = output[idx * d_model + n];
        scalar_t Pn_minus_1 = output[idx * d_model + (n - 1)];
        output[idx * d_model + (n + 1)] = ((2.0 * n + 1.0) * x * Pn - n * Pn_minus_1) / static_cast<scalar_t>(n + 1);
    }
}

std::vector<torch::Tensor> legendre_fused_forward_cuda(
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "legendre_fused_forward_cuda", ([&] {
        legendre_fused_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            max_seq_len,
            d_model,
            total_elements
        );
    }));

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }

    return {output};
}
