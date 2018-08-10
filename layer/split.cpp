//
// Created by 陈其友 on 2018/7/30.
//

#include "split.h"

namespace fastnn{

    Split::Split()
    {
        support_inplace=true;
    }

    int Split::infershape()
    {
        int number=tops.size();
        for(int i=0;i<number;i++)
        {
            tops[i]->setSize(bottoms[0]->h,bottoms[0]->w,bottoms[0]->c);
        }
        return 0;
    }

    int Split::forward() const
    {
        return 0;
    }
}
