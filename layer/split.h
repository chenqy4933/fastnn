//
// Created by 陈其友 on 2018/7/30.
//

#ifndef FASTNN_SPLIT_H
#define FASTNN_SPLIT_H

#include "layer.h"

namespace fastnn {

    class Split :public Layer{
    public:
        Split();

        virtual int infershape() override;

        int forward() const override;

    };

    Layer *GetSplitLayer() {
        return (Layer *)new Split();
    }
}


#endif //FASTNN_SPLIT_H
