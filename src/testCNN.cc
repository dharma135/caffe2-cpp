
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

#include "util/utils.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

namespace caffe2 {
  void print(Blob* blob, const std::string& name) {
	//auto tensor = blob->->Get<TensorCPU>();
	Tensor* tensor = BlobGetMutableTensor(blob, caffe2::DeviceType::CPU);
	const auto& data = tensor->data<float>();
	std::cout << name << "(" << tensor->sizes()
            				<< "): " << std::vector<float>(data, data + tensor->size())
							<< std::endl;
}
  void run() {
  std::cout << " Hello World! " <<std::endl;
  std::cout << " Testing GEMM from Caffe2" << std::endl;
  Workspace workspace;
	WorkspaceUtil wsUtil(workspace);
  DeviceOption option;
  caffe2::DeviceType device;
  CPUContext cpu_context(option);
  SmartTensorPrinter tprinter;

  device = CPU;
  std::vector<float> data(16*10);
  std::vector<int> dimData({16,10});
  int count = 0;
  for (auto& v : data) {
    v = (float)rand() / RAND_MAX;
    count++;
  }
  std::vector<int> label(16);
  std::vector<int> dimLabel({16,1});
  for (auto& v: label) {
    v = 16 * rand() / RAND_MAX;
  } 
  auto dataTen = Tensor(dimData, device);
	wsUtil.FeedBlob<float>("data",data,dimData,CPU);
  auto labelTen = Tensor(dimLabel, device);
	wsUtil.FeedBlob<int>("label",label,dimLabel,CPU);
  

  NetDef initModel;
	auto initNetUtil = NetUtil(initModel,"FC net Test");
  NetDef predictModel;
  auto predictNetUtil = NetUtil(predictModel,"FC predict");
	//auto modelUtil = ModelUtil(initModel,predictModel,"FCTest");
	//modelUtil.AddFcOps("data","fc",16,10,true);
 	initNetUtil.AddXavierFillOp(std::vector<int>{16,10}, "fc_w");
  // >>> bias = m.param_initModel.ConstantFill([], 'fc_b', shape=[10, ])
	initNetUtil.AddConstantFillOp(std::vector<int>{16}, "fc_b");

  std::vector<OperatorDef*> gradient_ops;
  auto fc_1 = predictNetUtil.AddFcOp("data","fc_w","fc_b","fc1");
	gradient_ops.push_back(fc_1);
	// >>> fc_1 = m.net.FC(["data", "fc_w", "fc_b"], "fc1")

  // >>> pred = m.net.Sigmoid(fc_1, "pred")
	auto pred = predictNetUtil.AddOp("Sigmoid",{"fc1"},{"pred"});
	gradient_ops.push_back(pred);

  // >>> [softmax, loss] = m.net.SoftmaxWithLoss([pred, "label"], ["softmax",
	// "loss"])
	auto loss = predictNetUtil.AddOp("SoftmaxWithLoss",{"pred","label"},{"softmax","loss"});
	gradient_ops.push_back(loss);
  // >>> m.AddGradientOperators([loss])
  auto loss_grad = predictNetUtil.AddOp("ConstantFill",{"loss"},{"loss_grad"});
	auto arg = loss_grad->add_arg();
	arg->set_name("value");
	arg->set_f(1.0);
	loss_grad->set_is_gradient_op(true);

  std::reverse(gradient_ops.begin(), gradient_ops.end());
	for (auto op : gradient_ops) {
		vector<GradientWrapper> output(op->output_size());
		for (auto i = 0; i < output.size(); i++) {
			output[i].dense_ = op->output(i) + "_grad";
		}
		GradientOpsMeta meta = GetGradientForOp(*op, output);
		auto grad = predictModel.add_op();
		grad->CopyFrom(meta.ops_[0]);
		grad->set_is_gradient_op(true);
	}

  // >>> print(str(m.net.Proto()))
	// std::cout << std::endl;
	// print(predictModel);

	// >>> print(str(m.param_init_net.Proto()))
	// std::cout << std::endl;
	// print(initModel);

	// >>> workspace.RunNetOnce(m.param_init_net)
	CAFFE_ENFORCE(workspace.RunNetOnce(initModel));


	// >>> workspace.CreateNet(m.net)
	CAFFE_ENFORCE(workspace.CreateNet(predictModel));


	// >>> for j in range(0, 100):
	for (auto i = 0; i < 100; i++) {
		// >>> data = np.random.rand(16, 100).astype(np.float32)
		//std::vector<float> data(16 * 100);
		std::vector<float> data(16*10);
		for (auto& v : data) {
			v = (float)rand() / RAND_MAX;
		}
		// >>> label = (np.random.rand(16) * 10).astype(np.int32)
		std::vector<int> label(16);
		for (auto& v : label) {
			v = rand() %10;
		}

		// >>> workspace.FeedBlob("data", data)
		wsUtil.FeedBlob<float>("data",data,dimData,CPU);

		// >>> workspace.FeedBlob("label", label)
    wsUtil.FeedBlob<int>("label",label,dimLabel,CPU);


		std::cout<<predictModel.DebugString()<<std::endl;
		std::cout<<predictModel.external_input_size()<<std::endl;
		predictModel.InitAsDefaultInstance();


		// >>> workspace.RunNet(m.name, 10)   # run for 10 times
		for (auto j = 0; j < 10; j++) {
			predictModel.CheckInitialized();
			CAFFE_ENFORCE(workspace.RunNet(predictModel.name()));
			 std::cout << "step: " << i << " loss: ";
			 print(workspace.GetBlob("loss"),"loss");
			 std::cout << std::endl;
		}

  }

}

}
int main(int argc, char ** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();
  caffe2::run();
  caffe2::ShutdownProtobufLibrary();
  return 0;
}