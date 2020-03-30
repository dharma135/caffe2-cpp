#include "util/workspace.h"
#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include "caffe2/core/blob.h"
#include <caffe2/core/operator.h>
#include "caffe2/core/tensor.h"

namespace caffe2 {
  Tensor WorkspaceUtil::FetchBlob(const std::string& name, DeviceType device) {
    auto tensor = BlobGetTensor(*ws.GetBlob(name),device).Clone();
    return tensor;
  }

}