#include<iostream>

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

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

uchar* ConvertMatToArray(Mat a) 
{
    vector<uchar> tmp;
    tmp.assign((uchar*)a.datastart, (uchar*)a.dataend);
    return &tmp[0];
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

Vec3b Convolution(const Mat src, vector<vector<float> >* kernel, int x, int y) 
{
    int k = (*kernel).size() / 2;
    int x_tmp, y_tmp;
    Vec3b sum = 0.0;
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

Mat ApplyKernelSequential(const Mat src, vector<vector<float> >* kernel) 
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

Mat ApplyKernelSequentialFlat(const Mat& src, vector<vector<float> >* kernel) 
{
    int kernel_flat_size = (*kernel).size() * (*kernel).size();
    int src_flat_size = src.rows * src.cols * src.elemSize();

    uchar* src_flat = ConvertMatToArray(src);
    float* kernel_flat = Convert2DVectorToArray(kernel);
    uchar* dst_flat = new uchar[src_flat_size];

    uchar sum = 0;
    int i_tmp;

    for(int i = 0; i < src_flat_size; i++)
    {
        for(int j = 0; j < kernel_flat_size; j++)
        {
            i_tmp = i; // TODO compute the right index on flat structures
            sum = sum + kernel_flat[j] * src_flat[i_tmp];
        }
        dst_flat[i] = sum;
    }
    
    Mat dst = Mat(src.rows, src.cols, src.type(), dst_flat);
    return dst;
}

Mat ApplyKernelParallel(const Mat src, vector<vector<float> >* kernel, int num_threads)
{
    Mat dst = Mat(src.rows, src.cols, src.type());
    omp_set_nested(1);
    int x,y;
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
 
int main(int argc, char *argv[])
{ 
    char* filename = argv[1];

    Mat src = imread(filename, CV_LOAD_IMAGE_COLOR);

    if (!src.data) 
    {
        cout << "Image file " << filename << "not found!" << endl;
        return -1;        
    }     

    cout << "Processing image file " << filename << endl;

    GaussianKernel kernel(5);

    Mat dst_sequential, dst_sequential_flat,
        dst_parallel, dst_parallel_flat;

    profile_omp_time__("*** Sequential version")
    {
        dst_sequential = ApplyKernelSequential(src, &(kernel.values));
    }

    profile_omp_time__("*** Sequential version - Flat")
    {
        dst_sequential_flat = ApplyKernelSequentialFlat(src, &(kernel.values));
    }
    cout << "    Correct: " << AreMatsEqual(dst_sequential, dst_sequential_flat) << endl;

    for(int t = 0; t < 4; t++) {
        int num_threads = pow(2, t);
        ostringstream label;
        label << "*** OpenMP version - " << num_threads << " thread(s)";
        profile_omp_time__(label.str())
        {
            dst_parallel = ApplyKernelParallel(src, &(kernel.values), num_threads);
        }
        cout << "    Correct: " << AreMatsEqual(dst_sequential, dst_parallel) << endl;
    }

    namedWindow("final", WINDOW_NORMAL);
    imshow("final", dst_sequential_flat);

    namedWindow("initial", WINDOW_NORMAL);
    imshow("initial", src);
 
    waitKey();
 
    return 0;
}