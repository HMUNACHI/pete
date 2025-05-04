# setup.py

import os
import sys # Add sys import
from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available() and cpp_extension.CUDA_HOME is not None

# Adjust source paths relative to the root directory
sources = ['polynomial_embeddings/index.cpp']
define_macros = []
extra_compile_args = {'cxx': ['-O3', '-std=c++17']} # Basic optimization and specify C++ standard

if cuda_available:
    print("CUDA detected, building with CUDA support.")
    # Adjust paths for CUDA files
    sources.extend([
        'polynomial_embeddings/chebyshev_kernel.cu',
        'polynomial_embeddings/fourier_kernel.cu',
        'polynomial_embeddings/legendre_kernel.cu',
        'polynomial_embeddings/hermite_kernel.cu',
        'polynomial_embeddings/laguerre_kernel.cu'
    ])
    define_macros += [('WITH_CUDA', None)]
    extra_compile_args['nvcc'] = ['-O3', '-arch=sm_70'] # Example: Optimize for Volta+
else:
    print("CUDA not detected, building with CPU support only.")
    # Add the CPU kernel source files
    sources.extend([
        'polynomial_embeddings/chebyshev_kernel.cpp',
        'polynomial_embeddings/fourier_kernel.cpp',
        'polynomial_embeddings/legendre_kernel.cpp',
        'polynomial_embeddings/hermite_kernel.cpp',
        'polynomial_embeddings/laguerre_kernel.cpp'
    ])
    # Add OpenMP flags for CPU parallelization only if not on macOS
    if sys.platform != 'darwin':
        # These flags might need adjustment based on the compiler (GCC/Clang vs MSVC)
        extra_compile_args['cxx'].append('-fopenmp') # For GCC/Clang
        # extra_compile_args['cxx'].append('/openmp') # For MSVC

setup(
    name='polynomial_embeddings',
    version='0.1.0', # Keeping version, can be adjusted if needed
    ext_modules=[
        cpp_extension.CppExtension(
            # Build as a top-level module instead of inside the package
            name='_polynomial_embeddings_C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    packages=['polynomial_embeddings'], # Package name remains the same
    # package_dir is removed as setup.py is now in the root
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
