//
// Created by 陈其友 on 2018/7/24.
//

#include "relu.h"

namespace fastnn{

    Relu::Relu()
    {

        support_inplace = true;
    }

    int Relu::load_param(const ParamDict& pd)
    {
        slope = pd.get(0, 0.f);

        return 0;
    }

    int Relu::forward() const
    {
        return 0;
    }
}
