

#ifndef FASTNN_NET_H
#define FASTNN_NET_H

#include <stdio.h>
#include <vector>
#include <map>
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

    int organize_net(void);
    //Forward the net

    //forward compute the whole network
    //return 0 if success
    int Forward(void);

    //optimize the whole network
    //return 0 if success
    //do the fuse of layer and so on
    int net_optimize();

    //plan for the memory
    //return the size of memory to be used
    //plan for the memory
    size_t net_memory_plan(int level);

    std::map<std::string,Blob*> input;
    std::map<std::string,Blob*> output;
    std::map<std::string,Blob> allBlob;
    std::vector<Layer*> allLayer;

    std::vector<float*> allPtr;
};

class Engine
{
public:
    // enable light mode
    // intermediate blob will be recycled when enabled
    // enabled by default
    void set_light_mode(bool enable);

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
    bool lightmode;     //in this model the memory is applied and deleted in runtime.
    int num_threads;
};

} //

#endif //FASTNN_NET_H
