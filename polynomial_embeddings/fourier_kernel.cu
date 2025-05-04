// chebyshev_kernel.cu

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// CUDA kernel for fully parallel Fourier expansion
template <typename scalar_t>
__global__ void fourier_fused_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int max_seq_len,
    const int d_model,
    const int total_elements // batch_size * seq_len * d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose linear index into batch, sequence, and frequency indices
    int token_idx = idx / d_model; // Token index: batch * seq
    int freq_idx = idx % d_model; // Frequency index for this token

    // Normalize input: Map x to range [-1, 1]
    scalar_t x = 2.0 * (input[token_idx] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;

    // Compute Fourier term based on frequency index
    scalar_t freq = static_cast<scalar_t>((freq_idx / 2) + 1); // Frequency scaling
    if (freq_idx % 2 == 0) {
        // Even index -> sine term
        output[idx] = sin(freq * x * M_PI);
    } else {
        // Odd index -> cosine term
        output[idx] = cos(freq * x * M_PI);
    }
}

std::vector<torch::Tensor> fourier_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto total_elements = batch_size * seq_len * d_model;

    auto output = torch::zeros({batch_size, seq_len, d_model}, input.options());

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fourier_fused_kernel", ([&] {
        fourier_fused_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            max_seq_len,
            d_model,
            total_elements
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }

    return {output};
}