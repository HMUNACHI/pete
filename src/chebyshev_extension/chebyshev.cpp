// chebyshev.cpp

#include <torch/extension.h>
#include <vector>

// Declaration of the CUDA forward function
std::vector<torch::Tensor> chebyshev_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
);

// C++ interface
std::vector<torch::Tensor> chebyshev_fused_forward(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    return chebyshev_fused_forward_cuda(input, max_seq_len, d_model);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &chebyshev_fused_forward, "Chebyshev Fused Forward (CUDA)");
}
