#include <iostream>
#include <random>
#include "util/dac.h"

int main()
{
    caffe2::D2AR2R dacr2r(6,100E-3,1.0);
    caffe2::D2A * dac;
    dac = &dacr2r;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);//uniform distribution between 0 and 1
    float * data_in;
    float * data_out;
    int numData;
    numData=23;
    data_in = new float[numData];
    data_out = new float[numData];
    for (int n = 0; n < numData; ++n) {
      data_in[n] = dis(gen)*pow(2,8);
      std::cout << data_in[n] << std::endl;
    }
    dac->convert(data_in,data_out,numData);
    for (int n = 0; n < numData; ++n) {
      std::cout <<"float=" <<data_in[n]<<"," << "Voltage="<< data_out[n]<< std::endl;
    }    
    return 0;
} 