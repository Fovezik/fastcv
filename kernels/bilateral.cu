#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"


using uchar = unsigned char;

template <typename T>
__device__ inline T squared(T x) {
    return x * x;
}

__global__ void bilateralKernel(
    uchar *in_image, 
    uchar *out_image, 
    int width, 
    int height,
    int channels, 
    int radius, 
    float sigma_color, 
    float sigma_space
) {

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ uchar shared_memory[];

    int shared_w = blockDim.x + 2 * radius;
    int shared_h = blockDim.y + 2 * radius;

    int shared_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int shared_threads = shared_w * shared_h * channels;

    int start_column = blockIdx.x * blockDim.x - radius;
    int start_row = blockIdx.y * blockDim.y - radius;

    for (int i = shared_thread_id; i < shared_threads; i += blockDim.x * blockDim.y) {
        
        int current_channel = i % channels;
        int temp = i / channels;
        int current_x = temp % shared_w;
        int current_y = temp / shared_h;

        int img_x = start_column + current_x;
        int img_y = start_row + current_y;
        
        if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
            shared_memory[i] = in_image[(img_y * width + img_x) * channels + current_channel];
        } else shared_memory[i] = 0;
    }

    __syncthreads();

    if (column < width && row < height) {
        
        int center_x = threadIdx.x + radius;
        int center_y = threadIdx.y + radius;

        for (int curr_channel = 0; curr_channel < channels; curr_channel++) {
            
            float image_filtered = 0;
            float normalization_factor = 0;

            uchar center_val =  shared_memory[(center_y * shared_w + center_x) * channels + curr_channel];

            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {

                    int neighbor_column = center_x + j;
                    int neighbor_row = center_y + i;
                    
                    uchar neighbor_val = shared_memory[(neighbor_row * shared_w + neighbor_column) * channels + curr_channel];

                    float intensity_diff = static_cast<float>(neighbor_val) - static_cast<float>(center_val);
                    float gaussian_color= exp(-(squared<float>(intensity_diff)) / (2 * squared<float>(sigma_color)));
                    
                    float gaussian_distance = exp(-(squared<float>(i) + squared<float>(j)) / (2 * squared<float>(sigma_space)));

                    float curr_norm_factor = gaussian_distance * gaussian_color;

                    image_filtered += neighbor_val * curr_norm_factor;
                    normalization_factor += curr_norm_factor;
                }
            }
            out_image[(row * width + column) * channels + curr_channel] = static_cast<uchar>(image_filtered / normalization_factor);
        }
    }
}


torch::Tensor bilateral_filter(torch::Tensor img, int filter_size, float sigma_color, float sigma_space) {
    
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    int radius = filter_size / 2;

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto calc_shared_memory_size = [&](int radius, int channel) -> size_t {
        size_t shared_w = dimBlock.x + 2 * radius;
        size_t shared_h = dimBlock.y + 2 * radius;
        return shared_w * shared_h * channel * sizeof(uchar);
    };

    size_t shared_memory_size = calc_shared_memory_size(radius, channels);

    auto result = torch::empty(
        {height, width, channels}, 
        torch::TensorOptions().dtype(torch::kByte).device(img.device())
    );

    bilateralKernel<<<dimGrid, dimBlock, shared_memory_size, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<uchar>(),
        result.data_ptr<uchar>(),
        width, height, channels, radius, sigma_color, sigma_space
    );
    
    cudaDeviceSynchronize();
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}