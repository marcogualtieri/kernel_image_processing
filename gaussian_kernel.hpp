#include<iostream>

using namespace std;

class GaussianKernel 
{

public:
    vector<vector<float> > values;

    GaussianKernel(size_t size) 
    {
        vector<vector<float> > tmp(size, vector<float>(size));
        int half_size = size / 2;
        double sigma = 1.0;
        double r, s = 2.0 * sigma * sigma;
        double sum = 0.0;
        for (int x = -half_size; x <= half_size; x++)
        {
            for (int y = -half_size; y <= half_size; y++)
            {
                r = sqrt(x*x + y*y);
                tmp[x + half_size][y + half_size] = (exp(-(r*r)/s))/(M_PI * s);
                sum += tmp[x + half_size][y + half_size];
            }
        }
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                tmp[i][j] /= sum;
        values = tmp;
    }

    void Print() 
    {
        for (int i = 0; i < values.size(); i++) 
        {
            for (int j = 0; j < values.size(); j++) 
            {
                cout << values[i][j] << " ";
            }
            cout << endl;
        }
    }
};