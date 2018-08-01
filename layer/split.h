//
// Created by 陈其友 on 2018/7/30.
//

#ifndef FASTNN_SPLIT_H
#define FASTNN_SPLIT_H

#include "layer.h"

namespace fastnn {

    class Split :public Layer{
    public:
        ~Split(){};
        Split(){};

        virtual int load_param(const ParamDict& pd) override{ return 0;};

        virtual int load_model(const ModelBin& mb) override{return 0;};

        int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const override{return 0;};

    };

    Layer *GetSplitLayer() {
        return (Layer *)new Split();
    }
}


#endif //FASTNN_SPLIT_H
