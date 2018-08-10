//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_DROPOUT_H
#define FASTNN_DROPOUT_H

#include "layer.h"

namespace fastnn {
class Dropout :public Layer{

public:
    Dropout();

    virtual int load_param(const ParamDict& pd) override;

    int forward() const override;

public:
    float scale;
};

Layer *GetDropoutLayer()
{
    return (Layer *) new Dropout();
}

}


#endif //FASTNN_DROPOUT_H
