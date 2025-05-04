#include "laguerre_kernel.h"

// Only compile the contents of this file if CUDA is not available.
#ifndef WITH_CUDA

#include <torch/extension.h>
#include <vector>

// Conditionally include OpenMP only on non-Apple platforms
#ifndef __APPLE__
#include <omp.h>
#endif

// CPU implementation for Laguerre
std::vector<torch::Tensor> laguerre_fused_forward_cpu(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto output = torch::zeros({batch_size, seq_len, d_model}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "laguerre_fused_forward_cpu", ([&] {
        auto input_a = input.accessor<scalar_t, 2>();
        auto output_a = output.accessor<scalar_t, 3>();

        // Conditionally enable OpenMP pragma
        #ifndef __APPLE__
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                scalar_t x = 2.0 * (input_a[b][s] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;
                // L_0(x) = 1
                output_a[b][s][0] = 1.0;
                if (d_model > 1) {
                    // L_1(x) = 1 - x
                    output_a[b][s][1] = 1.0 - x;
                }
                // Recurrence: (n+1) * L_{n+1}(x) = (2n+1-x) * L_n(x) - n * L_{n-1}(x)
                for (int n = 1; n < d_model - 1; ++n) {
                    output_a[b][s][n + 1] = ((2.0 * n + 1.0 - x) * output_a[b][s][n] - n * output_a[b][s][n - 1]) / (n + 1.0);
                }
            }
        }
    }));
    return {output};
}

#endif // WITH_CUDA 