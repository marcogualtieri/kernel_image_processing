#include <iostream>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <omp.h>

#include "gaussian_kernel.hpp"

#ifndef profile_omp_time__             
#define profile_omp_time__(label) for(double blockTime = -1; (blockTime == -1 ? (blockTime = omp_get_wtime()) != -1 : false); cout << label << " " << omp_get_wtime()-blockTime << endl)
#endif
 
using namespace std;
using namespace cv;

int Reflect(int size, int p) 
{
    if (p < 0)
        return -p - 1;
    if (p >= size)
        return 2*size - p - 1;
    return p;
}

bool AreMatsEqual(Mat a, Mat b) 
{
    return equal(a.begin<Vec3b>(), a.end<Vec3b>(), b.begin<Vec3b>());
}

/* 1D version */

uchar* ConvertMatToArray(Mat a) 
{
    vector<uchar> tmp;
    tmp.assign((uchar*)a.datastart, (uchar*)a.dataend);
    uchar* array = new uchar[tmp.size()];
    for(int x = 0; x < tmp.size(); x++) {
        array[x] = tmp[x];
    }
    return array;
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

Mat ApplyKernelParallel1D(const Mat& src, vector<vector<float> >* kernel, int num_threads) 
{
    uchar* src_flat = ConvertMatToArray(src);
    int src_flat_size = src.rows * src.cols * 3;
    uchar* dst_flat = new uchar[src_flat_size];

    float* kernel_flat = Convert2DVectorToArray(kernel);

    int k = (*kernel).size() / 2;
    int kernel_width = (*kernel).size();

    int x_tmp, y_tmp;

    float sum_b = 0.0;
    float sum_g = 0.0;
    float sum_r = 0.0;

    int x, y, flat_b_index;
    int i, j, flat_kernel_index;

    #pragma omp parallel for num_threads(num_threads) private(x,y,sum_b,sum_g,sum_r,i,j,x_tmp,y_tmp,flat_b_index,flat_kernel_index)
    for (int m = 0; m < src_flat_size; m += 3) 
    {
        x = m % (src.cols * 3) / 3;
        y = m / (src.cols * 3);
        
        sum_b = 0.0;
        sum_g = 0.0;
        sum_r = 0.0;

        for (int n = 0; n < kernel_width * kernel_width; n++) 
        {
            i = n % kernel_width;
            j = n / kernel_width;
        
            x_tmp = Reflect(src.cols, x-(j-k));
            y_tmp = Reflect(src.rows, y-(i-k));

            flat_b_index = x_tmp * 3 + src.cols * 3 * y_tmp;
            flat_kernel_index = i + kernel_width * j;

            sum_b += kernel_flat[flat_kernel_index] * src_flat[flat_b_index];
            sum_g += kernel_flat[flat_kernel_index] * src_flat[flat_b_index+1];
            sum_r += kernel_flat[flat_kernel_index] * src_flat[flat_b_index+2];
        }
        dst_flat[m] = sum_b;
        dst_flat[m+1] = sum_g;
        dst_flat[m+2] = sum_r;
    }
    Mat dst = Mat(src.rows, src.cols, src.type(), dst_flat);
    return dst;
}

/* 2D version */

Vec3b Convolution(const Mat& src, vector<vector<float> >* kernel, int x, int y) 
{
    int k = (*kernel).size() / 2;
    int x_tmp, y_tmp;
    Vec3b sum = 0;
    for (int i = -k; i <= k; i++) 
    {
        for (int j = -k; j <= k; j++) 
        {
            x_tmp = Reflect(src.cols, x-j);
            y_tmp = Reflect(src.rows, y-i);
            sum = sum + (*kernel)[j+k][i+k] * src.at<Vec3b>(y_tmp,x_tmp);
        }
    }
    return sum;
}

Mat ApplyKernelSequential(const Mat& src, vector<vector<float> >* kernel) 
{
    Mat dst = Mat(src.rows, src.cols, src.type());
    for (int y = 0; y < src.rows; y++) 
    {
        for (int x = 0; x < src.cols; x++) 
        {
            dst.at<Vec3b>(y,x) = Convolution(src, kernel, x, y);
        }
    }
    return dst;
}

Mat ApplyKernelParallel(const Mat& src, vector<vector<float> >* kernel, int num_threads)
{
    Mat dst = Mat(src.rows, src.cols, src.type());
    int x,y;
    omp_set_nested(1);
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (y = 0; y < src.rows; y++) 
    {
        for (x = 0; x < src.cols; x++) 
        {
            dst.at<Vec3b>(y,x) = Convolution(src, kernel, x, y);
        }
    }
    return dst;
}

void ApplyKernelSingleBlock(const Mat& src, Mat& dst, vector<vector<float> >* kernel, int x_start, int y_start, int block_side_length) 
{
    int x, y;
    for (y = y_start; y < y_start + block_side_length; y++) 
    {
        if(y == src.rows) break;
        for (x = x_start; x < x_start + block_side_length; x++) 
        {
            if(x == src.cols) break;
            dst.at<Vec3b>(y,x) = Convolution(src, kernel, x, y);
        }
    }
}

Mat ApplyKernelParallelBlock(const Mat& src, vector<vector<float> >* kernel, int num_threads, int block_side_length)
{
    Mat dst = Mat(src.rows, src.cols, src.type());
    omp_set_nested(1);
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int y = 0; y < src.rows; y += block_side_length) 
    {
        for (int x = 0; x < src.cols; x += block_side_length) 
        {
            ApplyKernelSingleBlock(src, dst, kernel, x, y, block_side_length);
        }
    }
    return dst;
}
 
int main(int argc, char *argv[])
{ 
    /* parse input and prepare data */

    if (argc < 6)
    {
        cerr << "Provide following parameters: " << endl;
        cerr << " - Image file path" << endl;
        cerr << " - Size of Gaussian kernel" << endl;
        cerr << " - Number of threads" << endl;
        cerr << " - Length of blocks side" << endl;
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

    istringstream num_threads_ss(argv[3]);
    uint num_threads;
    if (!(num_threads_ss >> num_threads))
    {
        cerr << "Invalid number of threads: " << argv[3] << endl;
        return -1;
    }
    
    istringstream block_side_length_ss(argv[4]);
    uint block_side_length;
    if (!(block_side_length_ss >> block_side_length))
    {
        cerr << "Invalid blocks width: " << argv[4] << endl;
        return -1;
    }

    bool display_images = string(argv[5]) == "true";

    cout << "\nProcessing image file " << filename << endl;

    /* 2D versions */

    Mat dst_sequential, 
        dst_parallel, dst_parallel_blocks,
        dst_sequential_1d, dst_parallel_1d;

    profile_omp_time__("* Sequential version - Elapsed time")
    {
        dst_sequential = ApplyKernelSequential(src, &(kernel.values));
    }

    ostringstream label;
    label << "* OpenMP version - " << num_threads << " thread(s) - Elapsed time";
    profile_omp_time__(label.str())
    {
        dst_parallel = ApplyKernelParallel(src, &(kernel.values), num_threads);
    }

    label.str("");
    label << "* OpenMP version - " << num_threads << " thread(s) - "<< block_side_length << "x" << block_side_length << " blocks - Elapsed time";
    profile_omp_time__(label.str())
    {
        dst_parallel_blocks = ApplyKernelParallelBlock(src, &(kernel.values), num_threads, block_side_length);
    }

    /* 1D version */

    label.str("");
    label << "* OpenMP version - " << num_threads << " thread(s) - 1D array - Elapsed time";
    profile_omp_time__(label.str())
    {
        dst_parallel_1d = ApplyKernelParallel1D(src, &(kernel.values), num_threads);
    }
    
    /* display result */

    if(display_images)
    {
        namedWindow("Source", WINDOW_NORMAL);
        imshow("Source", src);

        namedWindow("Result", WINDOW_NORMAL);
        imshow("Result", dst_sequential);

        waitKey();
    }
 
    return 0;
}