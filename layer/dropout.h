//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_DROPOUT_H
#define FASTNN_DROPOUT_H

#include "layer.h"

namespace fastnn {
class Dropout :public Layer{

public:
    ~Dropout(){};
    Dropout(){};

    virtual int load_param(const ParamDict& pd) override{return 0;};

    virtual int load_model(const ModelBin& mb) override{return 0;};

    virtual int infershape() override{};

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const override{return 0;};


};

Layer *GetDropoutLayer()
{
    return (Layer *) new Dropout();
}

}


#endif //FASTNN_DROPOUT_H
