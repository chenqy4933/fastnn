

#ifndef FASTNN_NET_H
#define FASTNN_NET_H

#include <stdio.h>
#include <vector>
#include <map>
#include <set>
#include <stack>

#include "blob.h"
#include "layer.h"

namespace fastnn {

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

    //Init the net
    int InitNet(std::map<std::string,std::vector<int>> inputSize);

    int organize_net(void);
    //Forward the net

    //forward compute the whole network
    //return 0 if success
    int Forward(void);

    //all the steps before Forward and after loadmodel
    int before_Forward(void);

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
    std::map<std::string,Blob> allBlob;
    std::vector<Layer*> allLayer;

    std::map<std::string,int> blob2ptr;
    std::vector<float*> allPtr;
    float * conmom_ptr;
    size_t comom_size;


    bool organized=false;
};

class NetEngine
{
public:

    // set thread count for this extractor
    // this will overwrite the global setting
    // default count is system depended
    void set_num_threads(int num_threads);


    // set input by blob name
    // return 0 if success
    int input(const char* blob_name, const float * data);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, float * data );

private:
    Net* net;
};

} //

#endif //FASTNN_NET_H
