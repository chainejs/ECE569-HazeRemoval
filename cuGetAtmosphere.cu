#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include "mex.h"

__global__ void cuGetAtmosphere(double m, double n, double n_pixels,
                                double n_search_pixels, double *dark_vec, 
                                double* image_vec, double* indices,
                                double *d_atmosphere){
    float *accumulator = new float[3];
    float *atmosphere = new float[3];

    int tid = threadIdx.x + blockIdx.x * blockDim.x; 

    if (threadIdx.x <= 3){
        accumulator[tid] = 0;
    }

    if (tid < n_search_pixels){
        for (int idx = 0; idx < 3; idx++){
            int j = indices[tid];
            accumulator[idx] = accumulator[idx] + image_vec[j];
        }
    }

    __syncthreads();

    if (tid < 3){
        atmosphere[tid] = accumulator[tid] / n_search_pixels; 
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    
    dim3 blockDim(256);
    dim3 gridDim(1);
    
    int i = 0;

    double M = *mxGetPr(prhs[0]);
    double N = *mxGetPr(prhs[1]);
    double n_pixels = *mxGetPr(prhs[2]);
    double n_search_pixels = *mxGetPr(prhs[3]);
    double *dark_vec = mxGetPr(prhs[4]);
    double *image_vec = mxGetPr(prhs[5]);
    double *indices = mxGetPr(prhs[6]);    
    
    double *d_dark_vec, *d_image_vec, *h_atmosphere, *d_atmosphere;

    plhs[0] = mxCreateDoubleMatrix(3,1,mxREAL);

    h_atmosphere = mxGetPr(plhs[0]);

    h_atmosphere = (double *)malloc(3 * sizeof(double));

    cudaMalloc((void **)&d_dark_vec, sizeof(dark_vec) * sizeof(double));
    cudaMalloc((void **)&d_image_vec, sizeof(image_vec) * sizeof(double));
    cudaMalloc((void **)&d_atmosphere, 3 * sizeof(double));

    cudaMemcpy(d_dark_vec, dark_vec, sizeof(dark_vec) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_vec, image_vec, sizeof(image_vec) * sizeof(double), cudaMemcpyHostToDevice);

    cuGetAtmosphere<<<blockDim, gridDim>>>(M, N, n_pixels, n_search_pixels, d_dark_vec, d_image_vec, indices, d_atmosphere);

    cudaMemcpy(h_atmosphere, d_atmosphere, 3 * sizeof(double), cudaMemcpyDeviceToHost);


}