#pragma once

#include <torch/extension.h>
#include <vector>

#ifdef WITH_CUDA
// Declare CUDA function
std::vector<torch::Tensor> legendre_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
);
#else
// Declare CPU function
std::vector<torch::Tensor> legendre_fused_forward_cpu(
    torch::Tensor input,
    int max_seq_len,
    int d_model
);
#endif 