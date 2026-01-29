#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>

#include <nvtx3/nvToolsExt.h>

#include <cmath>

#include "utils.cuh"

// Define uchar
using uchar = unsigned char;

// Helper function to compute square of a number
template <typename T>
__host__ __device__ inline T squared(T x) {
    return x * x;
}

// Bilateral filter kernel
__global__ void bilateralKernel(
    uchar *in_image, // input image
    uchar *out_image, // output image
    float *color_weights, // precomputed color weights
    float *distance_weights, // precomputed distance weights
    int width, // image width
    int height, // image height
    int channels, // number of channels
    int radius, // filter radius
    int filter_size // size of the filter
) {
    // Shared memory allocation
    extern __shared__ uchar shared_memory[];

    // Calculate global thread coordinates
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate shared memory dimensions
    int shared_w = blockDim.x + 2 * radius;
    int shared_h = blockDim.y + 2 * radius;

    // Calculate shared memory thread ID and total threads
    int shared_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int shared_threads = shared_w * shared_h * channels;

    // Calculate starting coordinates for shared memory
    int start_column = blockIdx.x * blockDim.x - radius;
    int start_row = blockIdx.y * blockDim.y - radius;

    // Load data into shared memory
    for (int i = shared_thread_id; i < shared_threads; i += blockDim.x * blockDim.y) {
        
        int current_channel = i % channels;
        int temp = i / channels;
        int current_x = temp % shared_w;
        int current_y = temp / shared_w;

        int img_x = start_column + current_x;
        int img_y = start_row + current_y;
        
        if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
            shared_memory[i] = in_image[(img_y * width + img_x) * channels + current_channel];
        } else shared_memory[i] = 0;
    }

    // Synchronize threads to ensure shared memory is fully loaded
    __syncthreads();
    
    // Apply bilateral filter
    if (column < width && row < height) {
        
        // Calculate center position in shared memory
        int shared_center_x = threadIdx.x + radius;
        int shared_center_y = threadIdx.y + radius;

        // Iterate over each channel
        for (int curr_channel = 0; curr_channel < channels; curr_channel++) {

            // Initialize filtered value and normalization factor
            float pixel_filtered = 0;
            float normalization_factor = 0;

            // Get center pixel value
            uchar center_val = shared_memory[(shared_center_y * shared_w + shared_center_x) * channels + curr_channel];

            // Iterate over the filter window
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    
                    // Calculate distance weight index
                    int distance_index = (i + radius) * filter_size + (j + radius);
                    float gaussian_distance = distance_weights[distance_index];

                    // Get neighbor pixel value
                    int neighbor_column = shared_center_x + j;
                    int neighbor_row = shared_center_y + i;

                    // Access neighbor value from shared memory
                    uchar neighbor_val = shared_memory[(neighbor_row * shared_w + neighbor_column) * channels + curr_channel];
                    
                    // Calculate color weight
                    int intensity_diff = abs((int)neighbor_val - (int)center_val);
                    float gaussian_color = color_weights[intensity_diff];

                    // Combine weights and accumulate
                    float curr_norm_factor = gaussian_distance * gaussian_color;
                    
                    // Accumulate filtered value and normalization factor
                    pixel_filtered += neighbor_val * curr_norm_factor;
                    normalization_factor += curr_norm_factor;
                }
            }
            
            // Write the normalized filtered value to output image
            out_image[(row * width + column) * channels + curr_channel] = static_cast<uchar>(pixel_filtered / normalization_factor);
        }
    }
}

// Bilateral filter function
torch::Tensor bilateral_filter(torch::Tensor img, int filter_size, float sigma_color, float sigma_space) {
    
    nvtxRangePushA("bilateral_filter");

        nvtxRangePushA("input_validation");
            assert(img.device().type() == torch::kCUDA); // Ensure input is on CUDA device
            assert(img.dtype() == torch::kByte); // Ensure input is of type unsigned char
        nvtxRangePop();

        nvtxRangePushA("prepare_parameters");
            // Prepare parameters
            const int height = img.size(0); // image height
            const int width = img.size(1); // image width
            const int channels = img.size(2); // number of channels
            const int radius = filter_size / 2; // filter radius
            const int filter_size_squared = squared<int>(filter_size); // filter size squared
        nvtxRangePop();

        nvtxRangePushA("calculate_color_weights_thrust");
            // Precompute color weights using Thrust
            thrust::device_vector<float> thrust_color_weights(256);
            thrust::counting_iterator<int> color_iterator(0);
            thrust::transform(
                thrust::device,
                color_iterator, 
                color_iterator + 256,
                thrust_color_weights.begin(),
                [=] __host__ __device__ (int intensity_diff) {
                    return expf(-(squared<float>(intensity_diff)) / (2.0f * squared<float>(sigma_color)));
                }
            );
        nvtxRangePop();

        nvtxRangePushA("calculate_distance_weights_thrust");
            // Precompute distance weights using Thrust
            thrust::device_vector<float> thrust_distance_weights(filter_size_squared);
            thrust::tabulate(thrust::device, thrust_distance_weights.begin(), thrust_distance_weights.end(),
                [=] __host__ __device__ (int distance_index) {
                    int column = distance_index % filter_size - radius;
                    int row = distance_index / filter_size - radius;
                    return expf(-(squared<float>(row) + squared<float>(column)) / (2 * squared<float>(sigma_space)));
                }
            );
        nvtxRangePop();

        nvtxRangePushA("get_raw_pointers_from_thrust_vectors");
            // Get raw pointers from Thrust device vectors
            float* color_weights = thrust::raw_pointer_cast(thrust_color_weights.data());
            float* distance_weights = thrust::raw_pointer_cast(thrust_distance_weights.data());
        nvtxRangePop();

        nvtxRangePushA("calculate_grid_block_dimensions");
            // Calculate grid and block dimensions
            dim3 dimBlock = getOptimalBlockDim(width, height);
            dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));
        nvtxRangePop();

        nvtxRangePushA("calculate_shared_memory_size");
            // Calculate shared memory size needed
            auto calc_shared_memory_size = [&](int radius, int channel) -> int {
                int shared_w = dimBlock.x + 2 * radius;
                int shared_h = dimBlock.y + 2 * radius;
                return shared_w * shared_h * channel * sizeof(uchar);
            };
            int shared_memory_size = calc_shared_memory_size(radius, channels);
        nvtxRangePop();

        nvtxRangePushA("allocate_output_tensor");
            // Allocate output tensor
            auto result = torch::empty(
                {height, width, channels}, 
                torch::TensorOptions().dtype(torch::kByte).device(img.device())
            );
        nvtxRangePop();

        nvtxRangePushA("kernel_launch");
            // Launch bilateral filter kernel
            bilateralKernel<<<dimGrid, dimBlock, shared_memory_size, at::cuda::getCurrentCUDAStream()>>>(
                img.data_ptr<uchar>(),
                result.data_ptr<uchar>(),
                color_weights, distance_weights,
                width, height, channels, radius, filter_size
            );
        nvtxRangePop();

        nvtxRangePushA("synchronize_and_check");
            // Synchronize and check for kernel launch errors
            cudaDeviceSynchronize();
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        nvtxRangePop();

    nvtxRangePop();
    return result;
}