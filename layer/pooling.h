//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_POOLING_H
#define FASTNN_POOLING_H

#include "layer.h"

namespace fastnn {

class Pooling: public Layer {

public:
    ~Pooling();
    Pooling();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int infershape();

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const;


};

Layer *GetPoolingLayer() {
    return (Layer *) new Pooling();
}
}
#endif //FASTNN_POOLING_H
