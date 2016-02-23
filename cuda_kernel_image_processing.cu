#include <iostream>

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
	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if((x<width) && (y<height))
	{
		//Location of pixel in input
		const int pixel_tid = y * width * 3 + (3 * x);

		//printf("%d,%d - %d\n",x,y,pixel_tid);
		//printf("%d\n", pixel_tid);
		/*const uchar blue	= src[pixel_tid];
		const uchar green	= src[pixel_tid + 1];
		const uchar red		= src[pixel_tid + 2];
		*/
		/*dst[pixel_tid] = 0;
		dst[pixel_tid + 1] = 0;
		dst[pixel_tid + 2] = 0;*/

		int i,j,x_tmp,y_tmp,flat_b_index,flat_kernel_index;
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

Mat convolution(const Mat& host_src, vector<vector<float> >* kernel)
{
	Mat host_dst = Mat::zeros(host_src.rows, host_src.cols, host_src.type());

	const int kernel_size = (*kernel).size()*(*kernel).size();
	float* kernel_flat = Convert2DVectorToArray(kernel);

	// allocate device memory
	const int bytes = host_src.step * host_src.rows;
	uchar *dev_src, *dev_dst;
	float* dev_kernel;
	SAFE_CUDA_CALL(cudaMalloc(&dev_src, bytes),"CUDA - malloc failed");
	SAFE_CUDA_CALL(cudaMalloc(&dev_dst, bytes),"CUDA - malloc failed");
	SAFE_CUDA_CALL(cudaMalloc(&dev_kernel, kernel_size * sizeof(float)),"CUDA - malloc failed");

	// copy data from OpenCV structure to device memory
	SAFE_CUDA_CALL(cudaMemcpy(dev_src, host_src.ptr(), bytes, cudaMemcpyHostToDevice),"CUDA - Memcpy host to device failed");
	SAFE_CUDA_CALL(cudaMemcpy(dev_kernel, kernel_flat, kernel_size * sizeof(float), cudaMemcpyHostToDevice),"CUDA - Memcpy host to device failed");

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
	const dim3 grid((host_src.cols + block.x - 1)/block.x, (host_src.rows + block.y - 1)/block.y);

	//Launch the color conversion kernel
	convolution_kernel<<<grid, block>>>(dev_src, dev_dst, host_src.cols, host_src.rows, host_src.step, dev_kernel, (*kernel).size());

	//Synchronize to check for any kernel launch errors
	SAFE_CUDA_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	//Copy back data from destination device memory to OpenCV output image
	SAFE_CUDA_CALL(cudaMemcpy(host_dst.ptr(),dev_dst,bytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CUDA_CALL(cudaFree(dev_src),"CUDA Free Failed");
	SAFE_CUDA_CALL(cudaFree(dev_dst),"CUDA Free Failed");
	SAFE_CUDA_CALL(cudaFree(dev_kernel),"CUDA Free Failed");

	return host_dst;
}

int main()
{

	string filename = "landscape.jpg";

	Mat src = imread(filename, CV_LOAD_IMAGE_COLOR);

	if(src.empty())
	{
		cout << "Image Not Found!" << endl;
		return -1;
	}

	GaussianKernel kernel(3);

	Mat dst = convolution(src, &(kernel.values));

	namedWindow("final", WINDOW_NORMAL);
	imshow("final", dst);

	namedWindow("initial", WINDOW_NORMAL);
	imshow("initial", src);

	waitKey();

	src.release();
	dst.release();

	SAFE_CUDA_CALL(cudaDeviceReset(),"CUDA can't clear memory");

	return 0;
}
