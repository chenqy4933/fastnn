//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_CONCAT_H
#define FASTNN_CONCAT_H

#include "layer.h"

namespace fastnn {

class Concat : public Layer {

public:
    ~Convolution();
    Convolution();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int infershape();

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const;

};


Layer *GetConcatLayer() {
    return (Layer *) new Concat();
}

}
#endif //FASTNN_CONCAT_H
