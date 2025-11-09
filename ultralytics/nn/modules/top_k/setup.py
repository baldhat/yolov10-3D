

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="topk_cuda",
    ext_modules=[
        CUDAExtension(
            name="topk_cuda",
            sources=["topk_cuda.cpp", "topk_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
