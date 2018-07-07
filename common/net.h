// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef FASTNN_NET_H
#define FASTNN_NET_H

#include <stdio.h>
#include <vector>
#include "blob.h"
#include "layer.h"
#include "mat.h"
#include "platform.h"

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
    void clear();

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

    std::map<std::string,blob*> input;
    std::map<std::string,blob*> output;
    std::map<std::string,blob> allBlob;
    std::vector<Layer*> allLayer;
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
    int input(const char* blob_name, const Mat& in);

    // get result by blob name
    // return 0 if success
    int extract(const char* blob_name, Mat& feat);

private:
    Net* net;
    bool lightmode;     //in this model the memory is applied and deleted in runtime.
    int num_threads;
};

} //

#endif //FASTNN_NET_H
