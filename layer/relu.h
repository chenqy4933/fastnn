//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_RELU_H
#define FASTNN_RELU_H

#include "layer.h"

namespace fastnn {

class Relu :public Layer{

public:
    ~Relu();
    Relu();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const;


};

Layer *GetReluLayer() {
    return (Layer *) new Relu();
}

}
#endif //FASTNN_RELU_H
