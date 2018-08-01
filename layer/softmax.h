//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_SOFTMAX_H
#define FASTNN_SOFTMAX_H


#include "layer.h"

namespace fastnn {

class Softmax: public Layer {

public:
    Softmax();
    ~Softmax(){};

    virtual int load_param(const ParamDict& pd) override ;

    virtual int load_model(const ModelBin& mb) override {return 0;};

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const override{return 0;};

public:
    int axis;
};

Layer *GetSoftmaxLayer() {
    return (Layer *) new Softmax();
}

}
#endif //FASTNN_SOFTMAX_H
