from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["/O2", "/std:c++17", "/permissive-"]
nvcc_args = [
   "-O2",
   "-std=c++17",
   "-D__CUDA_NO_HALF_OPERATORS__",
   "-D__CUDA_NO_HALF_CONVERSIONS__",
   "-D__CUDA_NO_HALF2_OPERATORS__",
   "-Xcompiler", "/std:c++17",
   "-Xcompiler", "/permissive-",
]

setup(
   name="fastcv",
   ext_modules=[
       CUDAExtension(
           name="fastcv",
           sources=[
               "kernels/grayscale.cu",
               "kernels/box_blur.cu",
               "kernels/sobel.cu",
               "kernels/dilation.cu",
               "kernels/erosion.cu",
               "kernels/module.cpp",
               "kernels/bilateral.cu"
           ],
           extra_compile_args={
               "cxx": cxx_args,
               "nvcc": nvcc_args,
           },
       ),
   ],
   cmdclass={"build_ext": BuildExtension},
)
