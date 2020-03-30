#ifndef __DATASET_H_
#define __DATASET_H_
#include <string>
namespace caffe2
{
  class dataset {
   public:
   std::string trainDBPath;
   std::string testDBPath;
   std::string dbType;
   
  };
} // namespace caffe2

#endif // __DATASET_H