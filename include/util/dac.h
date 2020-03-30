#ifndef __DAC_H_
#define __DAC_H_
#include <cmath>
namespace caffe2
{

  class D2A
  {
  protected:
    /* data */
    int nbits_;
    double vref_;
    double gain_;
    unsigned int nlevel_;
    double resolution_;
    
  public:
    D2A(int nbits,double vref, double gain=1.0E6)
    : nbits_(nbits), vref_(vref), gain_(gain) {

    };
    ~D2A() {};
    
    virtual void convert(float * data_in, float * data_out, int numData)=0;

  };

 class D2AR2R : public D2A {
   private:
   double * Vout_;
   public:
   D2AR2R(int nbits,double vref, double gain);
   void convert(float * data_in, float * data_out, int numData);
   ~D2AR2R() {
     delete[] Vout_;
   }
 };
  
} // namespace caffe2

#endif // __DAC_H_