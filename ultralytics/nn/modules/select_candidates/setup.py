from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="select_candidates_cuda",
    ext_modules=[
        CUDAExtension(
            name="select_candidates_cuda",
            sources=["select_candidates_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
