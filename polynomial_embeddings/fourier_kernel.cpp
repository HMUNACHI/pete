#include "fourier_kernel.h"

// Only compile the contents of this file if CUDA is not available.
#ifndef WITH_CUDA

#include <torch/extension.h>
#include <vector>
#include <cmath>

// Conditionally include OpenMP only on non-Apple platforms
#ifndef __APPLE__
#include <omp.h>
#endif

// CPU implementation for Fourier
std::vector<torch::Tensor> fourier_fused_forward_cpu(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto output = torch::zeros({batch_size, seq_len, d_model}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fourier_fused_forward_cpu", ([&] {
        auto input_a = input.accessor<scalar_t, 2>();
        auto output_a = output.accessor<scalar_t, 3>();
        #ifdef __APPLE__
        // Define M_PI if not available (e.g., strict C++) and not using OpenMP include
        const scalar_t pi = acos(-1.0);
        #else
        constexpr scalar_t pi = M_PI; // Use M_PI from cmath when omp.h might also include it
        #endif

        // Conditionally enable OpenMP pragma
        #ifndef __APPLE__
        #pragma omp parallel for collapse(3) schedule(static)
        #endif
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                 for (int n = 0; n < d_model; ++n) {
                    scalar_t x = 2.0 * (input_a[b][s] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;
                    scalar_t freq = static_cast<scalar_t>((n / 2) + 1);
                    if (n % 2 == 0) { // Even -> sine
                         output_a[b][s][n] = std::sin(freq * x * pi);
                     } else { // Odd -> cosine
                         output_a[b][s][n] = std::cos(freq * x * pi);
                     }
                 }
            }
        }
    }));
    return {output};
}

#endif // WITH_CUDA 