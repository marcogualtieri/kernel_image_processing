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

Vec3b Convolution(Mat src, vector<vector<float> >* kernel, int x, int y) 
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

Mat ApplyKernelSequential(Mat src, vector<vector<float> >* kernel) 
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

Mat ApplyKernelParallel(Mat src, vector<vector<float> >* kernel)
{
    Mat dst = Mat(src.rows, src.cols, src.type());
    omp_set_nested(1);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < src.rows; y++) 
    {
        for (int x = 0; x < src.cols; x++) 
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

    Mat dst_sequential, dst_parallel;

    profile_omp_time__("** Sequential version")
    {
        dst_sequential = ApplyKernelSequential(src, &(kernel.values));
    }

    profile_omp_time__("** Parallel version")
    {
        dst_parallel = ApplyKernelParallel(src, &(kernel.values));
    }

    cout << "** Results are equal: " << AreMatsEqual(dst_sequential, dst_parallel) << endl;

    namedWindow("final", WINDOW_NORMAL);
    imshow("final", dst_parallel);

    namedWindow("initial", WINDOW_NORMAL);
    imshow("initial", src);
 
    waitKey();
 
    return 0;
}