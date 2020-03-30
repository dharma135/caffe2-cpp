#include "util/dac.h"
#include <cmath>
#include <iostream>
namespace caffe2
{
  int getBit(int n, int k)
  {
      return (n & (1<<k)) != 0;
  }
  
D2AR2R::D2AR2R(int nbits,double vref, double gain=1.0E6) : D2A(nbits,vref,gain) {
     double base(2.0);
     double bi;
     resolution_ = vref_/std::pow(base,nbits);
     nlevel_ = std::pow(2,nbits_);
     Vout_ = new double[nlevel_];
     //(double *)malloc(nlevel_*sizeof(double));
     for (unsigned int ic=0; ic < nlevel_; ic++) {
       Vout_[ic] = 0.0;
       for (int kc=0; kc<nbits; kc++){
         bi = (double)getBit(ic,kc);
         Vout_[ic] = Vout_[ic] + bi*pow(base,kc);
       }
       Vout_[ic] *= gain_*resolution_;
     }
   };
  void D2AR2R::convert(float* data_in, float* data_out, int numData) {
    float voltage;
    bool isNegative;
    for (int ic=0; ic<numData; ic++) {
      isNegative = data_in[ic] < 0;
      int inx = (int)std::round(std::abs(data_in[ic]));
      if(inx > nlevel_) {
        inx = nlevel_ - 1;
      } 
      voltage = isNegative ? -1.0*Vout_[inx] : Vout_[inx];
      data_out[ic] = voltage;
      std::cout<<"---data="<<data_in[ic]<<",inx="<<inx<<",vout="<<Vout_[inx]<<",volt="<<voltage <<"---"<<std::endl;
    }
  }
} // namespace caffe2

