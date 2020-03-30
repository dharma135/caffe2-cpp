#ifndef CAFFE_OPERATORS_MULT5OP_H_
#define CAFFE_OPERATORS_MULT5OP_H_
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
namespace caffe2 {
  template <class Context>
  class Mult5Op final : public Operator<Context> {
  public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
    Mult5Op(const OperatorDef& operator_def, Workspace* ws)
     : Operator<Context>(operator_def,ws) , scale_(this->template GetSingleArgument<float>("scale", 1))
    {};
    bool RunOnDevice() override {
      return DispatchHelper<TensorTypes<int, int64_t, float, double>>::call(this,Input(DATA));
    }
    template <typename T>
    bool DoRunWithType();
  protected:
    INPUT_TAGS(DATA);
    float scale_;
  private:
  /* data */
  };

  template <class Context>
  class Mult5OpE final : public Operator<Context> {
  public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
    Mult5OpE(const OperatorDef& operator_def, Workspace* ws)
     : Operator<Context>(operator_def,ws), scale_(this->template GetSingleArgument<float>("scale", 1))
    {};
    bool RunOnDevice() override {
      return DispatchHelper<TensorTypes<int, int64_t, float, double>>::call(this,Input(DATA));
    }
    template <typename T>
    bool DoRunWithType();
  protected:
    INPUT_TAGS(DATA);
    float scale_;
  private:
  /* data */
  };

  template <class Context>
  class Mult5GradientOp final : public Operator<Context> {
  public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
    Mult5GradientOp(const OperatorDef& operator_def, Workspace* ws)
     : Operator<Context>(operator_def,ws)
    {};
    bool RunOnDevice() override {
      return DispatchHelper<TensorTypes<int, int64_t, float, double>>::call(this,Input(DATA));
    }
    template <typename T>
    bool DoRunWithType();
  protected:
    INPUT_TAGS(DATA);
  private:
  /* data */
  };


    
}
#endif //CAFFE_OPERATORS_MULT5OP