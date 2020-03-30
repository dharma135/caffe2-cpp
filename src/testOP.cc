#include <array>
#include <memory>
#include <vector>
#include <iostream>

#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/operator_gradient.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/smart_tensor_printer.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/context.h"
#include <gflags/gflags.h>
#include "util/utils.h"
#include "util/argparse.h"
#include "operators/mult5_op.h"
#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 { 
  void run() {
    std::cout << "Testing custom operator" << std::endl;
    Workspace ws;
    WorkspaceUtil wsUtil(ws);
    DeviceOption option;
    caffe2::DeviceType device;
    CPUContext cpu_context(option);
    SmartTensorPrinter tprinter;

    NetDef testModel;
    testModel.set_name("Test");
    {
    auto op = testModel.add_op();
    op->set_type("Mult5");
   // op->set_engine("XBAR");
    op->add_input("X");
    op->add_output("Y");
    auto arg = op->add_arg();
    arg->set_name("scale");
    arg->set_f(0.1f);
    }

    device = CPU;
    std::vector<int> dimX({16});
    std::vector<float> xv(16);
    for (auto& v : xv) {
      v = 1.0;
    }
  auto xTen = Tensor(dimX, device);
	wsUtil.FeedBlob<float>("X",xv,dimX,CPU);
  CAFFE_ENFORCE(ws.CreateNet(testModel));
    CAFFE_ENFORCE(ws.RunNet(testModel.name()));
  auto YTen = wsUtil.FetchBlob("Y",CPU);
  tprinter.Print(YTen);

  }
}
int main(int argc, char ** argv) {

  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();
  caffe2::run();
  caffe2::ShutdownProtobufLibrary();
  return 0;
}