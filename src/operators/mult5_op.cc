#include "mult5_op.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
template <typename T>
bool Mult5Op<CPUContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    // TODO - 1
    output_ptr[i] = data_ptr[i]*5*scale_;
  }
  return true;
}

template <>
template <typename T>
bool Mult5OpE<CPUContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    // TODO - 1
    output_ptr[i] = data_ptr[i]*5.2*scale_;
  }
  return true;
}

template <>
template <typename T>
bool Mult5GradientOp<CPUContext>::DoRunWithType() {
  const auto& data = Input(DATA);
  auto N = data.size();
  const auto* data_ptr = data.template data<T>();
  auto* output = Output(0);
  output->ResizeLike(data);
  auto* output_ptr = output->template mutable_data<T>();

  for (auto i = 0; i < N; i++) {
    // GI[0] = GO[0]
    // TODO - 2
    output_ptr[i] = data_ptr[i];
  }
  return true;
}

REGISTER_CPU_OPERATOR(Mult5, Mult5Op<CPUContext>);
REGISTER_CPU_OPERATOR_WITH_ENGINE(Mult5, XBAR ,Mult5OpE<CPUContext>);
OPERATOR_SCHEMA(Mult5)
    .NumInputs(1)
    .NumOutputs(1)
    .IdenticalTypeAndShape()
    .AllowInplace({{0, 0}})
    .SetDoc(R"DOC(
Element-wise add 5 operation. Each element in the output equals to the
corresponding element in the input data.

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "Add5",
    ["X"],
    ["Y"],
)

workspace.FeedBlob("X", (np.random.randint(100, size=(5,5))))
print("X before running op:", workspace.FetchBlob("X"))
workspace.RunOperatorOnce(op)
print("X after running op:", workspace.FetchBlob("Y"))

```

**Result**

```

X before running op:

X after running op:
[[6 2 3 3 0]
 [4 5 8 0 5]
 [4 6 4 3 6]
 [0 6 7 2 8]
 [1 4 6 7 5]]

```

 </details>

)DOC")
    .Input(0, "X", "Input tensor.")
    .Output(0, "Y", "Output tensor");
  //  .Arg("scale", "scale for multiplication");


REGISTER_CPU_OPERATOR(Mult5Gradient, Mult5GradientOp<CPUContext>);
OPERATOR_SCHEMA(Mult5Gradient)
    .NumInputs(1)
    .NumOutputs(1);

class GetMult5Gradient final : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Mult5Gradient",
        "",
        std::vector<std::string>{GO(0)},
        std::vector<std::string>{GI(0)});
  }
};

REGISTER_GRADIENT(Mult5, GetMult5Gradient);

} // namespace caffe2
