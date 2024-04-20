#include "mex.h" // Include MATLAB header file for MEX functions
#include "gpu/mxGPUArray.h" // Include MATLAB GPU Array header file
#include <cuda_runtime.h> // Include CUDA runtime API header file
#include <cmath> // Include math library for mathematical functions

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); } // Macro for checking CUDA errors

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
  if (code != cudaSuccess)
  {
    printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); // Print error message
    if (abort) exit(code); // Exit if abort is true
  }
}

void __global__ getTransmission(const float* darkChannel, float* transEst, float omega, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Calculate row index in the grid
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate column index in the grid

    if (row < m && col < n) { // Check if the thread is within image dimensions
        int index = (row * n + col); // Calculate 1D index from 2D indices
        // Compute transmission estimate using the given formula
        transEst[index] = 1.0f - omega * darkChannel[index];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Input variables
    mxGPUArray const *darkChannel = mxGPUCreateFromMxArray(prhs[0]); // Create GPU array from MATLAB input
    float omega = (float)(*mxGetPr(prhs[1])); // Get omega value from MATLAB input
    int m = (int)(*mxGetPr(prhs[2]));  // Get height of the input image from MATLAB input
    int n = (int)(*mxGetPr(prhs[3]));  // Get width of the input image from MATLAB input

    float const *d_darkChannel = (float *)(mxGPUGetDataReadOnly(darkChannel)); // Get pointer to dark channel data

    // Output variables
    mxGPUArray *transEst = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(darkChannel),
                                               mxGPUGetDimensions(darkChannel),
                                               mxGPUGetClassID(darkChannel),
                                               mxGPUGetComplexity(darkChannel),
                                               MX_GPU_DO_NOT_INITIALIZE); // Create GPU array for transmission estimation
    float *d_transEst = (float *)mxGPUGetData(transEst); // Get pointer to transmission estimation data

    // Calculate grid and block dimensions
    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y); // Calculate grid dimensions

    // Debugging information
    printf("Grid dimensions: %d x %d\n", blocksPerGrid.x, blocksPerGrid.y); // Print grid dimensions
    printf("Block dimensions: %d x %d\n", threadsPerBlock.x, threadsPerBlock.y); // Print block dimensions

    // Kernel launch
    printf("Launching kernel...\n"); // Print kernel launch message
    getTransmission<<<blocksPerGrid, threadsPerBlock>>>(d_darkChannel, d_transEst, omega, m, n); // Launch CUDA kernel for transmission estimation
    gpuErrchk(cudaPeekAtLastError()); // Check for kernel launch errors
    cudaDeviceSynchronize(); // Wait for kernel to finish and check for errors
    printf("Kernel execution completed.\n"); // Print kernel execution completion message

    // Create mxArray to return to MATLAB
    plhs[0] = mxGPUCreateMxArrayOnCPU(transEst); // Convert GPU array to MATLAB array and return

    // Clean up GPU memory
    mxGPUDestroyGPUArray(darkChannel); // Destroy dark channel GPU array
    mxGPUDestroyGPUArray(transEst); // Destroy transmission estimation GPU array
}
