# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="polynomial_embeddings",
    ext_modules=[
        CUDAExtension(
            name="polynomial_embeddings",
            sources=[
                "index.cpp",
                "chebyshev_kernel.cu",
                "legendre_kernel.cu",
            ],

        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
