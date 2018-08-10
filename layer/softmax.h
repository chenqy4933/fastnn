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

    virtual int load_param(const ParamDict& pd) override ;

    int forward() const override;

public:
    int axis;
};

Layer *GetSoftmaxLayer() {
    return (Layer *) new Softmax();
}

}
#endif //FASTNN_SOFTMAX_H
