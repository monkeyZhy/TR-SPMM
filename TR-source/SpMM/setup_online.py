from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='Libra6Online',
    ext_modules=[
        CUDAExtension(
            name='Libra6SpMMOnline',
            sources=[
                './mGCNkernel_online.cu',
                './mGCN_online.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'],
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
