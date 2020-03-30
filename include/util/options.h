#ifndef __OPTIONS_H_
#define __OPTIONS_H_
#include <string>

#define addopt3(type,name,helpstr,value) \
 type name=value; \
 std::string name ##_help=helpstr;

#define addopt2(type,name,helpstr) \
 type name; \
 std::string name ##_help=helpstr;

#define _GET_OVERRIDE(_1, _2, _3,_4,NAME, ...) NAME

#define addopt(...) _GET_OVERRIDE(__VA_ARGS__, \
    addopt3, addopt2)(__VA_ARGS__)

#define getOpt(x) x, x ## _help

namespace caffe2 {
  class options {
    public:
    // string opts
    addopt(std::string,model,"Name of the model")
    addopt(std::string,dataset,"Name of the dataset")
    addopt(std::string,db_type,"Type of DB","lmdb")
    addopt(std::string,data_dir,"Location of the data directory")
    addopt(std::string,model_dir,"Location of the model directory")
    // int opts
    int print_freq=25;
    addopt(int,num_iters,"Number of iterations for training",100)
    addopt(int,test_batch_size,"Batch size for tesitng",64)
    addopt(int,train_batch_size,"Batch size for training",64)
    int seed;
    int gpu; 
    // flags
    addopt(bool,eval,"if true evaluate model")
    //bool eval; 
    bool show;
    addopt(int,pretrained,"Loads Pretrained model")
    bool multiprocessing_distributed;
  
  };
} //namespace 
#endif