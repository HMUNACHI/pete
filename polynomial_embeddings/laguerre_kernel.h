#pragma once

#include <torch/extension.h>
#include <vector>

#ifdef WITH_CUDA
// Declare CUDA function
std::vector<torch::Tensor> laguerre_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len, // Still used for normalization consistency
    int d_model
);
#else
// Declare CPU function
std::vector<torch::Tensor> laguerre_fused_forward_cpu(
    torch::Tensor input,
    int max_seq_len, // Still used for normalization consistency
    int d_model
);
#endif 