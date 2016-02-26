#include <iostream>
#include <stdio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuda_runtime.h"

#include "gaussian_kernel.hpp"

using std::cout;
using std::cerr;
using std::endl;
using namespace cv;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		cerr << msg << endl << "File: " << file_name << endl << "Line number: " << line_number << endl << "Reason: " << cudaGetErrorString(err) << endl;
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CUDA_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

__device__ int Reflect(int size, int p)
{
    if (p < 0)
        return -p - 1;

    if (p >= size)
        return 2*size - p - 1;
    return p;
}

__global__ void convolution_kernel(uchar* src, uchar* dst, int width, int height, int width_step, float* kernel, int kernel_width)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	// consider only valid pixel coordinates
	if((x < width) && (y < height))
	{
		// pixel index in the src array
		const int pixel_tid = y * width * 3 + (3 * x);

		int i, j, x_tmp, y_tmp, flat_b_index, flat_kernel_index;
		int k = kernel_width / 2;
		float sum_b = 0.0;
		float sum_g = 0.0;
		float sum_r = 0.0;

		for (int n = 0; n < kernel_width*kernel_width; n++)
		{
			i = n % kernel_width;
			j = n / kernel_width;

			x_tmp = Reflect(width, x-(j-k));
			y_tmp = Reflect(height, y-(i-k));

			flat_b_index = x_tmp * 3 + width * 3 * y_tmp;
			flat_kernel_index = i + kernel_width * j;

			sum_b += kernel[flat_kernel_index] * src[flat_b_index];
			sum_g += kernel[flat_kernel_index] * src[flat_b_index+1];
			sum_r += kernel[flat_kernel_index] * src[flat_b_index+2];
		}
		
		dst[pixel_tid] = sum_b;
		dst[pixel_tid+1] = sum_g;
		dst[pixel_tid+2] = sum_r;
	}
}

float* Convert2DVectorToArray(vector<vector<float> >* kernel)
{
    int k = (*kernel).size();
    float* tmp = new float[k*k];
    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < k; j++)
        {
            tmp[j + i*k] = (*kernel)[i][j];
        }
    }
    return tmp;
}

void PrintDeviceProperties(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %zu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %zu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %zu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; i++)
	{
    	printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    	printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    }
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %zu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %zu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

Mat convolution(const Mat& host_src, vector<vector<float> >* kernel, int threads_block_size)
{
	Mat host_dst = Mat::zeros(host_src.rows, host_src.cols, host_src.type());

	const int kernel_size = (*kernel).size()*(*kernel).size();
	float* kernel_flat = Convert2DVectorToArray(kernel);

	// allocate device memory
	const int bytes = host_src.step * host_src.rows;
	uchar *dev_src, *dev_dst;
	float* dev_kernel;
	SAFE_CUDA_CALL(cudaMalloc(&dev_src, bytes),"CUDA malloc failed");
	SAFE_CUDA_CALL(cudaMalloc(&dev_dst, bytes),"CUDA malloc failed");
	SAFE_CUDA_CALL(cudaMalloc(&dev_kernel, kernel_size * sizeof(float)),"CUDA malloc failed");

	// copy data from OpenCV structure to device memory
	SAFE_CUDA_CALL(cudaMemcpy(dev_src, host_src.ptr(), bytes, cudaMemcpyHostToDevice),"CUDA memcpy host to device failed");
	SAFE_CUDA_CALL(cudaMemcpy(dev_kernel, kernel_flat, kernel_size * sizeof(float), cudaMemcpyHostToDevice),"CUDA memcpy host to device failed");

	// set block siza
	const dim3 block(threads_block_size, threads_block_size);
	// calculate grid size to cover the whole image
	const dim3 grid((host_src.cols + block.x - 1) / block.x, (host_src.rows + block.y - 1) / block.y);

	// Launch the color conversion kernel
	convolution_kernel<<<grid, block>>>(dev_src, dev_dst, host_src.cols, host_src.rows, host_src.step, dev_kernel, (*kernel).size());

	cudaError_t kernel_error = cudaGetLastError();
	if (kernel_error != cudaSuccess)
		cout << "CUDA Kernel Error: " << cudaGetErrorString(kernel_error) << endl;

	SAFE_CUDA_CALL(cudaDeviceSynchronize(),"CUDA Kernel Error");

	// copy data from device memory to OpenCV output image
	SAFE_CUDA_CALL(cudaMemcpy(host_dst.ptr(),dev_dst,bytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	// free the device memory
	SAFE_CUDA_CALL(cudaFree(dev_src),"CUDA Free Failed");
	SAFE_CUDA_CALL(cudaFree(dev_dst),"CUDA Free Failed");
	SAFE_CUDA_CALL(cudaFree(dev_kernel),"CUDA Free Failed");

	return host_dst;
}

int main(int argc, char *argv[])
{
	/* parse input and prepare data */

    if (argc < 5)
    {
        cerr << "Provide following parameters: " << endl;
        cerr << " - Image file path" << endl;
        cerr << " - Size of Gaussian kernel" << endl;
        cerr << " - Size of threads block" << endl;
        cerr << " - Display results (true/false)" << endl; 
        return -1;
    }

    char* filename = argv[1];
    Mat src = imread(filename, CV_LOAD_IMAGE_COLOR);
    if (!src.data) 
    {
        cerr << "Image file " << filename << "not found!" << endl;
        return -1;        
    }  

    istringstream gaussian_size_ss(argv[2]);
    uint gaussian_size;
    if (!(gaussian_size_ss >> gaussian_size) || (gaussian_size % 2) == 0)
    {
        cerr << "Gaussian kernel size must be an odd integer: " << argv[2] << endl;
        return -1;
    }
    GaussianKernel kernel(gaussian_size);

    istringstream threads_block_size_ss(argv[3]);
    uint threads_block_size;
    if (!(threads_block_size_ss >> threads_block_size))
    {
        cerr << "Invalid number of threads block size: " << argv[3] << endl;
        return -1;
    }

    bool display_images = string(argv[4]) == "true";

    // cout << "-------------------------" << endl;
    // cout << "--- DEVICE PROPERTIES ---" << endl;
    // cout << "-------------------------" << endl;
	// cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, 0);
    // PrintDeviceProperties(prop);

    cout << "\nProcessing image file " << filename << endl;

    /* apply convolution */

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	Mat dst = convolution(src, &(kernel.values), threads_block_size);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;

	cudaEventElapsedTime(&milliseconds, start, stop);

	cout << "* CUDA version - " << threads_block_size << "x" << threads_block_size << "x1 threads block - Elapsed time " << milliseconds << " ms" << endl; 

	/* display result */

    if(display_images)
    {
        namedWindow("Source", WINDOW_NORMAL);
        imshow("Source", src);

        namedWindow("Result", WINDOW_NORMAL);
        imshow("Result", dst);

        waitKey();
    }

    src.release();
	dst.release();

	SAFE_CUDA_CALL(cudaDeviceReset(),"Error in CUda device reset");
 
    return 0;
}
