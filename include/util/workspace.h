#ifndef __WORKSPACE_H_
#define __WORKSPACE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/init.h"
#include <caffe2/core/operator.h>
#include "caffe2/core/tensor.h"

namespace caffe2 {
class WorkspaceUtil {
  public:
  WorkspaceUtil (Workspace& ws) : ws(ws)  {

  }
  template<typename T>
  void FeedBlob(const std::string& name, std::vector<T>& data, const std::vector<int>& dim, DeviceType device);
  //template<typename T>
  Tensor FetchBlob(const std::string& name, DeviceType device);
  protected:
  Workspace& ws;
};
}
#endif // __WORKSPACE_H_

template <typename T>
void caffe2::WorkspaceUtil::FeedBlob(const std::string& name, std::vector<T>& data, const std::vector<int>& dim, DeviceType device)
{
  auto dataTen = Tensor(dim,device);
  auto tensor = BlobGetMutableTensor(ws.CreateBlob(name), device);
  auto ptr = dataTen.mutable_data<T>();
  int count = 0;
  for (auto& v: data) {
    ptr[count] = v;
    count++;
  }
  tensor->CopyFrom(dataTen);
}