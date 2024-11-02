# setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="chebyshev_extension",
    ext_modules=[
        CUDAExtension(
            name="chebyshev_extension",
            sources=["chebyshev.cpp", "chebyshev_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
