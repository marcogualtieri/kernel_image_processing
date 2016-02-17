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

void ProcessSingleBlock(const Mat& src, Mat& dst, vector<vector<float> >* kernel, int x_start, int y_start, int block_width) 
{
    int x, y;
    for (y = y_start; y < y_start + block_width; y++) 
    {
        if(y == src.rows) break;
        for (x = x_start; x < x_start + block_width; x++) 
        {
            if(x == src.cols) break;
            dst.at<Vec3b>(y,x) = Convolution(src, kernel, x, y);
        }
    }
}

Mat ApplyKernelParallelBlock(const Mat& src, vector<vector<float> >* kernel, int num_threads)
{
    Mat dst = Mat(src.rows, src.cols, src.type());
    int block_width = 64;
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int y = 0; y < src.rows; y+=block_width) 
    {
        for (int x = 0; x < src.cols; x+=block_width) 
        {
            ProcessSingleBlock(src, dst, kernel, x, y, block_width);
        }
    }
    return dst;
}
 
int main(int argc, char *argv[])
{ 
    if (argc < 5)
    {
        cerr << "Provide following parameters: " << endl;
        cerr << " - Image file path" << endl;
        cerr << " - Number of threads" << endl;
        cerr << " - Image blocks width" << endl;
        cerr << " - Display results (true/false)" << endl; 
        return -1;
    }

    char* filename = argv[1];

    istringstream num_threads_ss(argv[2]);
    int num_threads;
    if (!(num_threads_ss >> num_threads))
    {
        cerr << "Invalid number of threads: " << argv[2] << endl;
        return -1;
    }
    
    istringstream block_width_ss(argv[3]);
    int block_width;
    if (!(block_width_ss >> block_width))
    {
        cerr << "Invalid blocks width: " << argv[3] << endl;
        return -1;
    }

    bool display_images = string(argv[4]) == "true";

    Mat src = imread(filename, CV_LOAD_IMAGE_COLOR);

    if (!src.data) 
    {
        cout << "Image file " << filename << "not found!" << endl;
        return -1;        
    }     

    cout << "Processing image file " << filename << endl;

    GaussianKernel kernel(5);

    Mat dst_sequential, dst_parallel, dst_parallel_blocks;

    profile_omp_time__("*** Sequential version - Elapsed time")
    {
        dst_sequential = ApplyKernelSequential(src, &(kernel.values));
    }

    ostringstream label;
    label << "*** OpenMP version - " << num_threads << " thread(s) - Elapsed time";
    profile_omp_time__(label.str())
    {
        dst_parallel = ApplyKernelParallel(src, &(kernel.values), num_threads);
    }
    cout << "    Correct: " << AreMatsEqual(dst_sequential, dst_parallel) << endl;

    label.str("");
    label << "*** OpenMP version - " << num_threads << " thread(s) - Block width " << block_width << " - Elapsed time";
    profile_omp_time__(label.str())
    {
        dst_parallel_blocks = ApplyKernelParallelBlock(src, &(kernel.values), num_threads);
    }
    cout << "    Correct: " << AreMatsEqual(dst_sequential, dst_parallel_blocks) << endl;

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