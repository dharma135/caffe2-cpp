#ifndef __BLOB_H
#define __BLOB_H
#include "caffe2/core/blob.h"
#include "caffe2/core/tensor.h"
namespace caffe2{
class BlobUtil {
  public:
  BlobUtil(Blob* blob): blob_(blob) {}
  
  Tensor* Get();
  void Set(const TensorCPU &value, bool force_cuda = false);
  void Print(const std::string &name = "", int max = 100);
  protected:
  Blob* blob_;
};
}
#endif
