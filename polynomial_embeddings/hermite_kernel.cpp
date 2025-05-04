#include "hermite_kernel.h"

// Only compile the contents of this file if CUDA is not available.
#ifndef WITH_CUDA

#include <torch/extension.h>
#include <vector>

// Conditionally include OpenMP only on non-Apple platforms
#ifndef __APPLE__
#include <omp.h>
#endif

// CPU implementation for Hermite (Physicists')
std::vector<torch::Tensor> hermite_fused_forward_cpu(
    torch::Tensor input,
    int max_seq_len,
    int d_model
) {
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    auto batch_size = input.size(0);
    auto seq_len = input.size(1);
    auto output = torch::zeros({batch_size, seq_len, d_model}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hermite_fused_forward_cpu", ([&] {
        auto input_a = input.accessor<scalar_t, 2>();
        auto output_a = output.accessor<scalar_t, 3>();

        // Conditionally enable OpenMP pragma
        #ifndef __APPLE__
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                scalar_t x = 2.0 * (input_a[b][s] / static_cast<scalar_t>(max_seq_len - 1)) - 1.0;
                // H_0(x) = 1
                output_a[b][s][0] = 1.0;
                if (d_model > 1) {
                    // H_1(x) = 2x
                    output_a[b][s][1] = 2.0 * x;
                }
                // Recurrence: H_{n+1}(x) = 2x * H_n(x) - 2n * H_{n-1}(x)
                for (int n = 1; n < d_model - 1; ++n) {
                    output_a[b][s][n + 1] = 2.0 * x * output_a[b][s][n] - 2.0 * n * output_a[b][s][n - 1];
                }
            }
        }
    }));
    return {output};
}

#endif // WITH_CUDA 