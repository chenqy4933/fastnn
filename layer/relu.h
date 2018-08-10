//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_RELU_H
#define FASTNN_RELU_H

#include "layer.h"

namespace fastnn {

class Relu :public Layer{

public:

    Relu();

    virtual int load_param(const ParamDict& pd) override;


    int forward() const override;

public:
    float slope;
};

Layer *GetReluLayer() {
    return (Layer *) new Relu();
}

}
#endif //FASTNN_RELU_H
