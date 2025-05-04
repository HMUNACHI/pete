// index.cpp

#include <torch/extension.h>
#include <vector>

// Include kernel headers (declarations for both CPU and CUDA)
#include "chebyshev_kernel.h"
#include "fourier_kernel.h"
#include "legendre_kernel.h"
#include "hermite_kernel.h"
#include "laguerre_kernel.h"

// PYBIND11_MODULE definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
    // Bind CUDA functions
    m.def("chebyshev_fused_forward", &chebyshev_fused_forward_cuda, "Chebyshev Fused Forward (CUDA)");
    m.def("fourier_fused_forward", &fourier_fused_forward_cuda, "Fourier Fused Forward (CUDA)");
    m.def("legendre_fused_forward", &legendre_fused_forward_cuda, "Legendre Fused Forward (CUDA)");
    m.def("hermite_fused_forward", &hermite_fused_forward_cuda, "Hermite Fused Forward (CUDA)");
    m.def("laguerre_fused_forward", &laguerre_fused_forward_cuda, "Laguerre Fused Forward (CUDA)");
#else
    // Bind CPU functions
    m.def("chebyshev_fused_forward", &chebyshev_fused_forward_cpu, "Chebyshev Fused Forward (CPU)");
    m.def("fourier_fused_forward", &fourier_fused_forward_cpu, "Fourier Fused Forward (CPU)");
    m.def("legendre_fused_forward", &legendre_fused_forward_cpu, "Legendre Fused Forward (CPU)");
    m.def("hermite_fused_forward", &hermite_fused_forward_cpu, "Hermite Fused Forward (CPU)");
    m.def("laguerre_fused_forward", &laguerre_fused_forward_cpu, "Laguerre Fused Forward (CPU)");
#endif
}