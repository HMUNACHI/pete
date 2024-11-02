// chebyshev_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for fused normalization and Chebyshev expansion
template <typename scalar_t>
__global__ void chebyshev_fused_kernel(
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

    // Initialize Chebyshev terms
    // Assuming output is stored as [batch, seq, d_model]
    // Each thread handles one [batch, seq] position
    output[idx * d_model + 0] = 1.0; // T0(x) = 1
    if (d_model > 1) {
        output[idx * d_model + 1] = x; // T1(x) = x
    }
    for (int n = 2; n < d_model; ++n) {
        output[idx * d_model + n] = 2.0 * x * output[idx * d_model + (n - 1)] - output[idx * d_model + (n - 2)];
    }
}

std::vector<torch::Tensor> chebyshev_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    // Ensure input is contiguous and on CUDA
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto total_elements = batch_size * seq_len;

    // Allocate output tensor
    auto output = torch::zeros({batch_size, seq_len, d_model}, input.options());

    // Launch kernel
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "chebyshev_fused_forward_cuda", ([&] {
        chebyshev_fused_kernel<scalar_t><<<blocks, threads>>>(
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
