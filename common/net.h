

#ifndef FASTNN_NET_H
#define FASTNN_NET_H

#include <stdio.h>
#include <vector>
#include <map>
#include <set>
#include <stack>

#include "blob.h"
#include "layer.h"
#include "layer/input.h"

namespace fastnn {

class NetEngine;

class Net
{
public:
    // empty init
    Net();
    // clear and destroy
    ~Net();

    // load network structure from binary param file
    // return 0 if success
    int load_param(FILE* fp);
    int load_param(const char* protopath);

    // load network weight data from model file
    // return 0 if success
    int load_model(FILE* fp);
    int load_model(const char* modelpath);

    // unload network structure and weight data
    int clear();

    //clone the Net used the same weight but reallocate other computer blob
    Net* clone();

    int serialize_net(void);
    //Forward the net

    int find_input_output(void);

    //forward compute the whole network
    //return 0 if success
    int Forward(void);

    //all the steps before Forward and after loadmodel
    int InitNet(void);

    //optimize the whole network
    //return 0 if success
    //do the fuse of layer and so on
    int net_optimize();

    int fuse_layer(Layer* baseLayer,Layer * next);

    //plan for the memory
    //return the size of memory to be used
    //plan for the memory
    int net_memory_plan(void);

    int set_input_size(std::map<std::string,std::vector<int>> sizeOfinput);

    int memory_alloc(void);

    std::map<std::string,Blob*> input;
    std::map<std::string,Blob*> output;
    std::map<std::string,Blob*> allBlob;
    std::vector<Layer*> allLayer;

    std::map<std::string,int> blob2ptr;
    std::vector<int> ptrSize;
    std::vector<float*> allPtr;

    float * conmom_ptr=NULL;
    size_t comom_size=0;

    bool organized=false;

    friend class NetEngine;
};

class NetEngine
{
public:

    NetEngine(const char* param_path,const char* model_path);

    ~NetEngine();
    int init(void);

    int forward(void);

    int  get_input_shape(int *ptr, const char* name);

    int set_input_shape(int *ptr,const char* name);
    // set input by blob name
    // return 0 if successs
    int input(const char* blob_name, Blob& in_blob);

    // get result by blob name
    // return 0 if success
    int output(const char* blob_name, Blob& out_blob);

private:
    Net* net;
    bool inited=false;
    std::map<std::string,std::vector<int>> inputsize;

};

} //

#endif //FASTNN_NET_H
