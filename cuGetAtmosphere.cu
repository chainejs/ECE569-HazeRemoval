#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include "/usr/local/MATLAB/R2024a/toolbox/parallel/gpu/extern/include/gpu/mxGPUArray.h"

void __global__ cuGetAtmosphere(double m, double n, double n_pixels,
                                double n_search_pixels, double const *dark_vec, 
                                double const *image_vec, double const *indices,
                                double *d_atmosphere){
    __shared__ float shared_accum;
    double *accumulator = new double[3];

    // Set all three accumulator fields to 0 along with each blocks privitized version of the shared_accum variable
    if (threadIdx.x == 0 && blockIdx.x < 3){
        accumulator[blockIdx.x] = 0;
        shared_accum = 0;
    }

    __syncthreads();

    // Thread 0 from each block does the accumulation using an atomic add
    if (threadIdx.x == 0 && blockIdx.x < 3){
        for (int idx = 0; idx < n_search_pixels; idx++){
            int j = indices[idx] + (blockIdx.x * n_pixels);
            atomicAdd(&shared_accum, image_vec[j]);
        }
    }

    __syncthreads();

    // Thread 0 from each block will write their blocks respective privitized shared_accum to the global accumulator
    if (threadIdx.x == 0 && blockIdx.x < 3){
        d_atmosphere[blockIdx.x] = shared_accum / n_search_pixels; 
    }
}

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    // Block and Grid dimensions
    int const blockD = 512;
    int gridD = 30;

    // This section gets the inputs to the kernel from MATLAB and puts them into
    // a form that the GPU device side can use.
    double M = *mxGetPr(prhs[0]);
    double N = *mxGetPr(prhs[1]);
    double n_pixels = *mxGetPr(prhs[2]);
    double n_search_pixels = *mxGetPr(prhs[3]);

    mxGPUArray const *dark_vec = mxGPUCreateFromMxArray(prhs[4]);
    mxGPUArray const *image_vec = mxGPUCreateFromMxArray(prhs[5]);
    mxGPUArray const *indices = mxGPUCreateFromMxArray(prhs[6]);
    mxGPUArray const *atmo = mxGPUCreateFromMxArray(prhs[7]);
    
    double const *d_dark_vec = (double *)(mxGPUGetDataReadOnly(dark_vec)); 
    double const *d_image_vec = (double *)(mxGPUGetDataReadOnly(image_vec));

    // Create a MATLAB GPU array to hold the resulting atmosphere output
    mxGPUArray *h_atmosphere = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(atmo),
                                                    mxGPUGetDimensions(atmo),
                                                    mxGPUGetClassID(atmo),
                                                    mxGPUGetComplexity(atmo),
                                                    MX_GPU_DO_NOT_INITIALIZE);

    // Creat a device side pointer to hold the data from the host side MATLAB GPU array
    double *d_atmosphere = (double *)mxGPUGetData(h_atmosphere);
    double const *d_indices = (double *)(mxGPUGetDataReadOnly(indices));

    mxInitGPU();

    // Call the kernel for atmospheric detection
    cuGetAtmosphere<<<gridD, blockD>>>(M, N, n_pixels, n_search_pixels, d_dark_vec, d_image_vec, d_indices, d_atmosphere);
    printf("\n%d\n", cudaPeekAtLastError());

    // Write the code back to a form that MATLAB can use for atmosphere output on its side
    plhs[0] = mxGPUCreateMxArrayOnGPU(h_atmosphere);

    // Destroy all created mxGPUArrays
    mxGPUDestroyGPUArray(h_atmosphere);
    mxGPUDestroyGPUArray(dark_vec);
    mxGPUDestroyGPUArray(image_vec);
    mxGPUDestroyGPUArray(indices);
    mxGPUDestroyGPUArray(atmo);
}