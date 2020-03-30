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
#include "util/options.h"
#include "CLI/CLI.hpp"
#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

C10_DEFINE_string(MLDB,"/project/SAS_xfer/data_hub/reddy/Work2019/MLNN/MLDB/data/","MLDatabase path");
C10_DEFINE_string(train_db, "/project/SAS_xfer/data_hub/reddy/Work2019/MLNN/MLDB/data/caffe2/mnist/mnist-train-nchw-leveldb","The given path to the training leveldb.");
C10_DEFINE_string(test_db, "/project/SAS_xfer/data_hub/reddy/Work2019/MLNN/MLDB/data/caffe2/mnist/mnist-test-nchw-leveldb","The given path to the testing leveldb.");
C10_DEFINE_int(iters, 100, "The of training runs.");
C10_DEFINE_int(test_runs, 50, "The of test runs.");
C10_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");
C10_DEFINE_bool(display, false, "Display graphical training info.");

namespace caffe2 {
// >> def AddInput(model, batch_size, db, db_type):
void AddInput(ModelUtil &model, int batch_size, const std::string &db,
              const std::string &db_type, bool useTensorProto=true) {
  // Setup database connection
  model.init.AddCreateDbOp("dbreader", db_type, db);
  model.predict.AddInput("dbreader");
  
  // >>> data_uint8, label = model.TensorProtosDBInput([], ["data_uint8",
  // "label"], batch_size=batch_size, db=db, db_type=db_type)
  //AddTensorProtosDbInputOp
  if (useTensorProto) { 
      model.predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",
                                         batch_size);
  }
  else {
  model.predict.AddImageInputOp("dbreader", "data_uint8", "label",
                                         batch_size);
  }


  // >>> data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
  model.predict.AddCastOp("data_uint8", "data_f", TensorProto_DataType_FLOAT);

  // >>> data = model.Scale(data, data, scale=float(1./256))
  model.predict.AddScaleOp("data_f", "datag", 1.f / 256);

  // >>> data = model.StopGradient(data, data)
  model.predict.AddStopGradientOp("datag","data");
}

// def AddLeNetModel(model, data):
void AddLeNetModel(ModelUtil &model, bool test, std::string order="NCHW") {
  // >>> conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
  model.AddConvOps("data", "conv1", 1, 20, 1, 0, 5, 1, order, test);
  
  // >>> pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
  model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2,order);

  // >>> conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50,
  // kernel=5)
  model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, 1, order, test);

  // >>> pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
  model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2,order);

  // >>> fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
  model.AddFcOps("pool2", "fc3", 800, 500, test);

  // >>> fc3 = brew.relu(model, fc3, fc3)
  model.predict.AddReluOp("fc3", "fc3");

  // >>> pred = brew.fc(model, fc3, 'pred', 500, 10)
  model.AddFcOps("fc3", "pred", 500, 10, test);

  // >>> softmax = brew.softmax(model, pred, 'softmax')
  model.predict.AddSoftmaxOp("pred", "softmax");
}

// def AddAccuracy(model, softmax, label):
void AddAccuracy(ModelUtil &model) {
  // >>> accuracy = model.Accuracy([softmax, label], "accuracy")
  model.predict.AddAccuracyOp("softmax", "label", "accuracy");

  if (FLAGS_display) {
    model.predict.AddTimePlotOp("accuracy");
  }

  // >>> ITER = model.Iter("iter")
  model.AddIterOps();
}

// >>> def AddTrainingOperators(model, softmax, label):
void AddTrainingOperators(ModelUtil &model) {
  // >>> xent = model.LabelCrossEntropy([softmax, label], 'xent')
  model.predict.AddLabelCrossEntropyOp("softmax", "label", "xent");

  // >>> loss = model.AveragedLoss(xent, "loss")
  model.predict.AddAveragedLossOp("xent", "loss");

  if (FLAGS_display) {
    model.predict.AddShowWorstOp("softmax", "label", "data", 256, 0);
    model.predict.AddTimePlotOp("loss");
  }

  // >>> AddAccuracy(model, softmax, label)
  AddAccuracy(model);

  // >>> model.AddGradientOperators([loss])
  model.predict.AddConstantFillWithOp(1.0, "loss", "loss_grad");
  model.predict.AddGradientOps();

  // >>> LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
  // stepsize=1, gamma=0.999 )
  model.predict.AddLearningRateOp("iter", "LR", 0.1);

  // >>> ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1],
  // value=1.0)
  model.init.AddConstantFillOp({1}, 1.f, "ONE");
  model.predict.AddInput("ONE");

  // >>> for param in model.params:
  for (auto param : model.Params()) {
    // >>> param_grad = model.param_to_grad[param]
    // >>> model.WeightedSum([param, ONE, param_grad, LR], param)
    model.predict.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"},
                                   param);
  }

  // Checkpoint causes problems on subsequent runs
  // >>> model.Checkpoint([ITER] + model.params, [],
  // std::vector<std::string> inputs({"iter"});
  // inputs.insert(inputs.end(), params.begin(), params.end());
  // model.predict.AddCheckpointOp(inputs, 20, "leveldb",
  //                         "mnist_lenet_checkpoint_%05d.leveldb");
}

// >>> def AddBookkeepingOperators(model):
void AddBookkeepingOperators(ModelUtil &model) {
  // >>> model.Print('accuracy', [], to_file=1)
  model.predict.AddPrintOp("accuracy", true);

  // >>> model.Print('loss', [], to_file=1)
  model.predict.AddPrintOp("loss", true);

  // >>> for param in model.params:
  for (auto param : model.Params()) {
    // >>> model.Summarize(param, [], to_file=1)
    model.predict.AddSummarizeOp(param, true);

    // >>> model.Summarize(model.param_to_grad[param], [], to_file=1)
    model.predict.AddSummarizeOp(param + "_grad", true);
  }
}

  void run(caffe2::options opt) {

  std::cout << std::endl;
  std::cout << "## Caffe2 MNIST Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/tutorial-MNIST.html" << std::endl;
  std::cout << std::endl;
  std::string trainDB;
  std::string testDB;
  std::string order;
  //
  trainDB = FLAGS_MLDB+"caffe2/mnist/mnist-train-nchw-"+opt.db_type;
  testDB = FLAGS_MLDB+"caffe2/mnist/mnist-test-nchw-"+opt.db_type;
  
  std::cout << "train-db: " << trainDB << std::endl;
  std::cout << "test-db: " << testDB << std::endl;
  std::cout << "iters: " << opt.num_iters << std::endl;
  std::cout << "test-runs: " << FLAGS_test_runs << std::endl;
  std::cout << "force-cpu: " << (FLAGS_force_cpu ? "true" : "false")
            << std::endl;
  std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;
  // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize,
  // brew
  // >>> workspace.ResetWorkspace(root_folder)
  Workspace workspace("./tmp");
  WorkspaceUtil wsutil2(workspace);
  // >>> train_model = model_helper.ModelHelper(name="mnist_train",
  // arg_scope={"order": "NCHW"})
  order = "NCHW";
  std::string db_type;
  int train_batch_size=opt.train_batch_size;
  int test_batch_size=opt.test_batch_size;
  db_type = opt.db_type;
  // Create Training model
  if(!opt.pretrained) {
  NetDef train_init_model, train_predict_model;
  ModelUtil train(train_init_model, train_predict_model, "mnist_train");
  AddInput(train,train_batch_size, trainDB, db_type);
  AddLeNetModel(train, false,order);
  AddTrainingOperators(train);
  AddBookkeepingOperators(train);
  std::cout <<"---- Begin Training ----"<< std::endl;  
  CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));
  CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));
  for (auto i = 1; i <= opt.num_iters; i++) {
    CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));
    if (i % 10 == 0) {
      auto accuracyTen = wsutil2.FetchBlob("accuracy",CPU);
      auto accuracy = accuracyTen.data<float>()[0];
      auto loss = wsutil2.FetchBlob("loss",CPU).data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << accuracy << std::endl;
    }
  }
  std::cout <<"---- End Training ----"<< std::endl;
  train.predict.WriteText("./tmp/mnist-train_predict_net.pbtxt");
  train.predict.WriteGraph("./tmp/train-predict");
  train.init.WriteText("./tmp/mnist-train_init_net.pbtxt");  
  train.Write("tmp/mnist-train"); 
  } 
  // Create Testing model
  NetDef test_init_model, test_predict_model;
  ModelUtil test(test_init_model, test_predict_model,"mnist_test");
  //AddInput(test,test_batch_size, testDB, db_type);
  AddInput(test,test_batch_size, testDB, db_type);
    std::cout << "saving model input.. (./tmp/mnist_%_net.pb)" << std::endl;
  test.predict.WriteText("./tmp/inmnist-test_predict_net.pbtxt");
  test.predict.WriteGraph("./tmp/intest-predict");
  test.init.WriteText("./tmp/inmnist-test_init_net.pbtxt");
  test.Write("tmp/inmnist-test");  
  if (opt.pretrained) {
    NetDef dep_init_model, dep_predict_model;
    ModelUtil dep(dep_init_model, dep_predict_model,"mnist_dep");    
    dep.Read("./tmp/mnist");
    test.predict.net.MergeFrom(dep.predict.net);
    test.init.net.MergeFrom(dep.init.net);
    /*CAFFE_ENFORCE(workspace.RunNetOnce(dep.init.net));
    CAFFE_ENFORCE(workspace.CreateNet(dep.predict.net));
    CAFFE_ENFORCE(workspace.RunNet(dep.predict.net.name()));
    test.CopyDeploy(dep,workspace);*/
    //test.Read("./tmp/mnist");
    //AddInput(test,test_batch_size, testDB, db_type);
    //test.predict.Read("./tmp/mnist_predict_net.pb");
  std::cout << "saving model.. (./tmp/mnist_%_net.pb)" << std::endl;
  test.predict.WriteText("./tmp/pmnist-test_predict_net.pbtxt");
  test.predict.WriteGraph("./tmp/ptest-predict");
  test.init.WriteText("./tmp/pmnist-test_init_net.pbtxt");
  test.Write("tmp/pmnist-test");  
  } else {
    AddLeNetModel(test, true,order);
  }
  AddAccuracy(test);
  CAFFE_ENFORCE(workspace.RunNetOnce(test.init.net));
  CAFFE_ENFORCE(workspace.CreateNet(test.predict.net));

  std::cout << "--- Begin Testing ----" << std::endl;
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    CAFFE_ENFORCE(workspace.RunNet(test.predict.net.name()));
    if (i % 10 == 0) {
      auto accuracy = wsutil2.FetchBlob("accuracy",CPU).data<float>()[0];
      std::cout << "step: " << i << " accuracy: " << accuracy << std::endl;
    }
  }
  if (opt.pretrained) {
  std::cout << "saving model.. (./tmp/mnist_%_net.pb)" << std::endl;
  test.predict.WriteText("./tmp/mnist-test_predict_net.pbtxt");
  test.predict.WriteGraph("./tmp/test-predict");
  test.init.WriteText("./tmp/mnist-test_init_net.pbtxt");
  test.Write("tmp/mnist-test");     
  }
  std::cout <<"---- End Testing ----"<< std::endl;
  if(!opt.pretrained) {
  // Create Deployment Model
  NetDef deploy_init_model, deploy_predict_model;
  ModelUtil deploy(deploy_init_model, deploy_predict_model, "mnist_model");
  //deploy.predict.AddInput("dbreader");
  //deploy.predict.AddOutput("softmax");
  AddLeNetModel(deploy, true);
 for (auto &param : deploy.predict.net.external_input()) {
    auto tensor = wsutil2.FetchBlob(param,CPU);
    auto op = deploy.init.net.add_op();
    op->set_type("GivenTensorFill");
    auto arg1 = op->add_arg();
    arg1->set_name("shape");
    for (auto d : tensor.sizes()) {
      arg1->add_ints(d);
    }
    auto arg2 = op->add_arg();
    arg2->set_name("values");
    auto data = tensor.data<float>();
    for (auto i = 0; i < tensor.size(); i++) {
      arg2->add_floats(data[i]);
    }
    op->add_output(param);
  }

  std::cout << std::endl;
  std::cout << "saving model.. (./tmp/mnist_%_net.pb)" << std::endl;
  deploy.predict.WriteText("./tmp/mnist_predict_net.pbtxt");
  deploy.init.WriteText("./tmp/mnist_init_net.pbtxt");
  deploy.Write("tmp/mnist"); 
  }
  /*std::cout << "training..labels" << std::endl;
  SmartTensorPrinter tprinter;
   CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));
   auto lab = wsutil2.FetchBlob("label",CPU);
   auto dat = wsutil2.FetchBlob("data",CPU);
   std::cout << "labelSize" << lab.sizes() << std::endl;
   std::cout << "data Size" << dat.sizes() << std::endl;
   tprinter.Print(lab);
   */
    // >>> for i in range(total_iters):




  // with open(os.path.join(root_folder, "deploy_net.pbtxt"), 'w') as fid:
  // fid.write(str(deploy_model.net.Proto()))
  
  }
  void AddOptions(CLI::App& app, caffe2::options& opt) {
     app.add_option("-m,--model",opt.model,opt.model_help);
     app.add_option("-d,--dataset",opt.dataset,opt.dataset_help);
     app.add_option("--db_type",getOpt(opt.db_type));
     app.add_option("--data_dir",opt.data_dir,opt.data_dir_help);
     app.add_option("--model_dir",opt.model_dir,opt.model_dir_help);
     app.add_option("-e,--eval",getOpt(opt.eval));
     app.add_option("--num_iters",getOpt(opt.num_iters));
     app.add_flag("--pretrained",getOpt(opt.pretrained));
     app.set_help_flag("--help_mnist","Print this help message and exit");
  }
}
int main(int argc, char ** argv) {
  //caffe2::ArgumentParser parser;
  caffe2::options opt;
  CLI::App app{"Caffe2 test MNIST app"};
  app.allow_extras(true);
  caffe2::AddOptions(app,opt);
  CLI11_PARSE(app, argc, argv);
  //std::string str = app.config_to_str(false, true);
  std::cout << "sucessfully parsed options" << std::endl;
  std::cout << opt.db_type << "pretrained" << opt.pretrained << std::endl;
 // char** argv2;
 // int argc2;
  gflags::AllowCommandLineReparsing();
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();
  caffe2::run(opt);
  caffe2::ShutdownProtobufLibrary();
  return 0;
}