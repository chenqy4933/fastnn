//
// Created by 陈其友 on 2018/7/24.
//

#include "softmax.h"

namespace fastnn {

    Softmax::Softmax()
    {
        support_inplace = true;
    }

    int Softmax::load_param(const ParamDict& pd)
    {
        axis = pd.get(0, 0);

        return 0;
    }
}