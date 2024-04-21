
#include <cuda_runtime.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cmath>



__global__ void getRadianceKernel(double const *image, double const *max_transmission, double const *rep_atmosphere, double* radiance, int m, int n) {

    // create matrix index for row and column
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // create the shared memory array that will be used to store the image array
     __shared__ double sh_arr[1920][2560][3];


    //check the boundary conditions
        if (idx < n && idy < m) {
          int index = idy*n+idx;
          
    //iterate through each channel and calculate the radiance
        for (int k = 0; k < 3; k++) {
            sh_arr[idy][idx][k] = image[index*3+k];
          //  radiance[index*3+k] = ((image[index*3+k] - rep_atmosphere[k]) / max_transmission[index*3+k]) + rep_atmosphere[k];

           radiance[index*3+k] = ((sh_arr[idy][idx][k] - rep_atmosphere[k]) / max_transmission[index*3+k]) + rep_atmosphere[k];


        
        }

        __syncthreads();
    
    }
} //end of getRadianceKernel


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check input and output arguments
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("getRadiance:nrhs", "six inputs required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("getRadiance:nlhs", "One output required.");
    }
   
    // create the expected input variables on the right hand side of the kernel
    mxGPUArray const *image = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *transmission = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const *atmosphere = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const *radiance = mxGPUCreateFromMxArray(prhs[3]);

    int  m = *mxGetPr(prhs[4]);
    int  n = *mxGetPr(prhs[5]);
  
    

    // Allocate Memory and Create output array

      double const *d_image = (double *)(mxGPUGetDataReadOnly(image));
      double const *d_transmission = (double *)(mxGPUGetDataReadOnly(transmission));
      double const *d_atmosphere = (double *)(mxGPUGetDataReadOnly(atmosphere));
      mxGPUArray *h_radiance = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(radiance),
                                                    mxGPUGetDimensions(radiance),
                                                    mxGPUGetClassID(radiance),
                                                    mxGPUGetComplexity(radiance),
                                                    MX_GPU_DO_NOT_INITIALIZE);
      double *d_radiance = (double *)mxGPUGetData(h_radiance);


     mxInitGPU();


    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    // Call CUDA kernel
    getRadianceKernel<<<gridSize, blockSize>>>(d_image, d_transmission, d_atmosphere, d_radiance, m, n);

    // Copy result back to host
   
    plhs[0] = mxGPUCreateMxArrayOnGPU(h_radiance);

    // Free GPU memory
    // This is the cudaFree equivalent
    
    mxGPUDestroyGPUArray(h_radiance);
    mxGPUDestroyGPUArray(image);
    mxGPUDestroyGPUArray(transmission);
    mxGPUDestroyGPUArray(atmosphere);
    mxGPUDestroyGPUArray(radiance);
}

