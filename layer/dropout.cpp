//
// Created by 陈其友 on 2018/7/24.
//

#include "dropout.h"

namespace fastnn {

    Dropout::Dropout()
    {
        support_inplace = true;
    }

    int Dropout::load_param(const ParamDict& pd)
    {
        scale = pd.get(0, 1.f);

        return 0;
    }

    int Dropout::forward() const
    {
        return 0;
    }
}