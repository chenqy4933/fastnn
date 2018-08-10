//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_POOLING_H
#define FASTNN_POOLING_H

#include "layer.h"

namespace fastnn {

class Pooling: public Layer {

public:
    Pooling();

    virtual int load_param(const ParamDict& pd) override;

    virtual int infershape() override;

    int forward() const override;

    enum { PoolMethod_MAX = 0, PoolMethod_AVE = 1 };
public:
    int pooling_type;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    int global_pooling;
    int pad_mode;// 0=full 1=valid 2=SAME

};

Layer *GetPoolingLayer() {
    return (Layer *) new Pooling();
}
}
#endif //FASTNN_POOLING_H
