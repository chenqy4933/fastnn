//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_CONCAT_H
#define FASTNN_CONCAT_H

#include "layer.h"

namespace fastnn {

class Concat : public Layer {

public:
    ~Concat(){};
    Concat(){};

    virtual int load_param(const ParamDict& pd) override{return 0;};

    virtual int load_model(const ModelBin& mb) override{return 0;};

    virtual int infershape() override {return 0;};

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const override{return 0;};

};


Layer *GetConcatLayer() {
    return (Layer *) new Concat();
}

}
#endif //FASTNN_CONCAT_H
