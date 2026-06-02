from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='Libra5Online',
    ext_modules=[
        CppExtension(
            name='Libra5BlockOnline',
            sources=[
                './Block/block_online.cpp',
            ],
            extra_compile_args=['-O3', '-fopenmp', '-mcx16'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
