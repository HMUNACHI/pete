// index.cpp

#include <torch/extension.h>
#include <vector>

// Declaration of the CUDA forward function
std::vector<torch::Tensor> chebyshev_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
);

std::vector<torch::Tensor> legendre_fused_forward_cuda(
    torch::Tensor input,
    int max_seq_len,
    int d_model
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chebyshev", &chebyshev_fused_forward_cuda, "Chebyshev Fused Forward (CUDA)");
    m.def("legendre", &legendre_fused_forward_cuda, "Legendre Fused Forward (CUDA)");
}
