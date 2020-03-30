#include <array>
#include <memory>
#include <vector>
#include <iostream>

#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/smart_tensor_printer.h"
#include "caffe2/core/logging.h"
#ifdef WITH_CUDA
#include "caffe2/core/context_gpu.h"
#endif
namespace caffe2 {
  void run() {
  std::cout << " Hello World! " <<std::endl;
  std::cout << " Testing GEMM from Caffe2" <<std::endl;
  DeviceOption option;
  CPUContext cpu_context(option);
  SmartTensorPrinter tprinter;
  auto X = Tensor(std::vector<int>{5, 10}, CPU);
  auto W = Tensor(std::vector<int>{10, 6}, CPU);
  auto Y = Tensor(std::vector<int>{5, 6}, CPU);
  LOG(INFO) << "Tensor created X on" << X.GetDeviceType();
  tprinter.PrintMeta(X);
  LOG(INFO) << "Tensor created W on" << X.GetDeviceType();
  LOG(INFO) << "Tensor created Y on" << X.GetDeviceType();
  //std::cout << option << std::endl;  
  // set values to X, W
  math::Set<float,CPUContext>(X.numel(),1,X.mutable_data<float>(),&cpu_context);
  math::Set<float,CPUContext>(W.numel(),1,W.mutable_data<float>(),&cpu_context);
  LOG(INFO) << "Intialized matrix X and Matrix W to values 1";
  LOG(INFO) << "Calling GEMM to compute  Y = W*X";
  auto xDims = X.sizes();
  auto wDims = W.sizes();
  //LOG(INFO) << xDims ;
  const float kOne = 1.0;
  const float kPointFive = 0.5;
  const float kZero = 0.0;
  math::Gemm<float,CPUContext>(
    CblasNoTrans,
    CblasNoTrans,
    xDims[0],
    wDims[1],
    xDims[1],
    kOne,
    X.data<float>(),
    W.data<float>(),
    kZero,
    Y.mutable_data<float>(),
    &cpu_context
    );
  LOG(INFO) << "Completed MAtrix multiplication";
  tprinter.Print(Y);
  }
}

int main(int argc, char **argv){
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();
  caffe2::run();
  return 0;
}