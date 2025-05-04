# Import the top-level C module and alias it
import _polynomial_embeddings_C as _C

# Expose the C++ functions at the package level
chebyshev = _C.chebyshev_fused_forward
fourier = _C.fourier_fused_forward
legendre = _C.legendre_fused_forward
hermite = _C.hermite_fused_forward
laguerre = _C.laguerre_fused_forward

__all__ = [
    'chebyshev',
    'fourier',
    'legendre',
    'hermite',
    'laguerre'
] 