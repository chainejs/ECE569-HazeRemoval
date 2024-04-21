#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
//#include "/usr/local/MATLAB/R2024a/toolbox/parallel/gpu/extern/include/gpu/mxGPUArray.h"
#include "gpu/mxGPUArray.h"

void __global__ cuGetAtmosphere(double m, double n, double n_pixels,
                                double n_search_pixels, double const *dark_vec, 
                                double const *image_vec, double const *indices,
                                double *d_atmosphere){
    __shared__ float shared_accum;
    double *accumulator = new double[3];
    
    if (threadIdx.x == 0 && blockIdx.x < 3){
        accumulator[blockIdx.x] = 0;
        shared_accum = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x < 3){
        for (int idx = 0; idx < n_search_pixels; idx++){
            int j = indices[idx] + (blockIdx.x * n_pixels);
            atomicAdd(&shared_accum, image_vec[j]);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x < 3){
        d_atmosphere[blockIdx.x] = shared_accum / n_search_pixels; 
    }
}

#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    int const blockD = 512;
    int gridD = 30;

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
    mxGPUArray *h_atmosphere = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(atmo),
                                                    mxGPUGetDimensions(atmo),
                                                    mxGPUGetClassID(atmo),
                                                    mxGPUGetComplexity(atmo),
                                                    MX_GPU_DO_NOT_INITIALIZE);
    double *d_atmosphere = (double *)mxGPUGetData(h_atmosphere);
    double const *d_indices = (double *)(mxGPUGetDataReadOnly(indices));

    mxInitGPU();

    cuGetAtmosphere<<<gridD, blockD>>>(M, N, n_pixels, n_search_pixels, d_dark_vec, d_image_vec, d_indices, d_atmosphere);
    printf("\n%d\n", cudaPeekAtLastError());

    plhs[0] = mxGPUCreateMxArrayOnGPU(h_atmosphere);

    mxGPUDestroyGPUArray(h_atmosphere);
    mxGPUDestroyGPUArray(dark_vec);
    mxGPUDestroyGPUArray(image_vec);
    mxGPUDestroyGPUArray(indices);
    mxGPUDestroyGPUArray(atmo);
}