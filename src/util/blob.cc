#include "util/blob.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/tensor.h"
#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {

Tensor* BlobUtil::Get() {
#ifdef WITH_CUDA
  if (blob_.IsType<TensorCUDA>()) {
    return TensorCPU(blob_.Get<TensorCUDA>());
  }
#endif
  return BlobGetMutableTensor(blob_,DeviceType::CPU);
  //blob_.GetMutable<TensorCPU>();
}

void BlobUtil::Set(const TensorCPU &value, bool force_cuda) {
#ifdef WITH_CUDA
  if (force_cuda || blob_.IsType<TensorCUDA>()) {
    auto tensor = blob_.GetMutable<TensorCUDA>();
    tensor->CopyFrom(value);
    return;
  }
#endif
  auto tensor = BlobGetMutableTensor(blob_,DeviceType::CPU);
  tensor->ResizeLike(value);
  tensor->ShareData(value);
}

/*void BlobUtil::Print(const std::string &name, int max) {
 auto tensor = Get();
  TensorUtil(tensor).Print(name, max);
}*/

}  // namespace caffe2