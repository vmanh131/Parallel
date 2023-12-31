#include "My_GPU.h"

#include <cuda_runtime.h>


// CUDA kernel for convolutionssss
// __global__ void conv_forward_gpu(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
// {
//     int b = blockIdx.x;
//     int m = blockIdx.y;
//     int h = threadIdx.x;
//     int w = threadIdx.y;

//     if (b < B && m < M && h < H_out && w < W_out)
//     {
//         float sum = 0;
//         for (int c = 0; c < C; c++)
//         {
//             for (int p = 0; p < K; p++)
//             {
//                 for (int q = 0; q < K; q++)
//                 {
//                     int x_index = ((b * C + c) * H + h + p) * W + w + q;
//                     int k_index = ((m * C + c) * K + p) * K + q;
//                     sum += x[x_index] * k[k_index];
//                 }
//             }
//         }
//         y[((b * M + m) * H_out + h) * W_out + w] = sum;
//     }
// }

// // Function to call the CUDA kernel
// __host__ void MyGPU::conv_forward_gpu_caller(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;

//     dim3 blocks(B, M);
//     dim3 threads(H_out, W_out);

//     conv_forward_gpu<<<blocks, threads>>>(y, x, k, B, M, C, H, W, K, H_out, W_out);

//     cudaDeviceSynchronize();
// }

// Define the constant memory for the filter
__constant__ float d_k_const[3200]; //MxCxKxK = 4x16x7x7

__global__ void conv_forward_gpu(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K, const int H_out, const int W_out)
{
    // Get the block and thread indices
    int bx = blockIdx.x; // block index along x-axis
    int by = blockIdx.y; // block index along y-axis
    int tx = threadIdx.x; // thread index within a block along x-axis

    // Calculate the output pixel coordinates
    int h = by; // output pixel row
    int w = bx; // output pixel column

    // Calculate the input image and output feature map indices
    int b = tx / M; // input image index
    int m = tx % M; // output feature map index

    // Initialize the output pixel value to zero
    float sum = 0.0f;

    // Loop over the filter coefficients
    for (int c = 0; c < C; c++)
    {
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                // Calculate the input pixel coordinates
                int i = h + p; // input pixel row
                int j = w + q; // input pixel column

                // Get the input pixel value
                float x_val = x[(b * C + c) * H * W + i * W + j];

                // Get the filter coefficient from constant memory
                float k_val = d_k_const[(m * C + c) * K * K + p * K + q];

                // Accumulate the product of the input pixel and the filter coefficient
                sum += x_val * k_val;
            }
        }
    }

    // Store the output pixel value
    y[(b * M + m) * H_out * W_out + h * W_out + w] = sum;
}

__host__ void MyGPU::conv_forward_gpu_caller(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Calculate the output image dimensions
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    // Allocate device memory for input and output
    float *d_x, *d_y;
    cudaMalloc(&d_x, B * C * H * W * sizeof(float));
    cudaMalloc(&d_y, B * M * H_out * W_out * sizeof(float));

    // Copy input from host to device
    cudaMemcpy(d_x, x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // Copy filter from host to constant memory
    cudaMemcpyToSymbol(d_k_const, k, M * C * K * K * sizeof(float));

    // Define the grid and block dimensions
    dim3 gridDim(W_out, H_out); // grid size is H_out * W_out
    dim3 blockDim(B * M, 1); // block size is B * M

    // Launch the kernel
    conv_forward_gpu<<<gridDim, blockDim>>>(y, x, B, M, C, H, W, K, H_out, W_out);

    // Copy output from device to host
    cudaMemcpy(y, d_y, B * M * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
}

