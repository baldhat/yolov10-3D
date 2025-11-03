# ------------------------------------------------------------
# setup.py
# ------------------------------------------------------------
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='sparse_conv2d',
    ext_modules=[
        CUDAExtension(
            name='sparse_conv2d',
            sources=['sparse_conv2d.cu'],
            extra_compile_args={'cxx': ['-O3'],
                                'nvcc': ['-O3',
                                         '--expt-relaxed-constexpr',
                                         '-arch=sm_60']}  # adjust SM version to your GPU
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)