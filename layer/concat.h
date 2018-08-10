//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_CONCAT_H
#define FASTNN_CONCAT_H

#include "layer.h"

namespace fastnn {

class Concat : public Layer {

public:
    Concat();

    virtual int load_param(const ParamDict& pd) override;

    virtual int infershape() override ;

    int forward() const override;

public:
    int axis=0;

};


Layer *GetConcatLayer() {
    return (Layer *) new Concat();
}

}
#endif //FASTNN_CONCAT_H
