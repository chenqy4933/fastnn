//
// Created by 陈其友 on 2018/7/24.
//

#include "concat.h"

namespace fastnn {

    Concat::Concat()
    {
        support_inplace = false;
    }

    int Concat::load_param(const ParamDict& pd)
    {
        axis = pd.get(0, 0);

        return 0;
    }

    int Concat::infershape()
    {
        int number=bottoms.size();
        if(axis==0)
        {
            int channel=bottoms[0]->c;
            for (int i = 1; i < number; i++)
            {
                if(bottoms[0]->h==bottoms[i]->h && bottoms[0]->w==bottoms[i]->w)
                    channel+=bottoms[i]->c;
                else {
                    printf("The cancat size is not correct!!\n");
                    return -1;
                }

            }
            tops[0]->setSize(bottoms[0]->h,bottoms[0]->w,channel);
            return 0;
        }

        if(axis==1)
        {
            int height=bottoms[0]->h;
            for (int i = 1; i < number; i++)
            {
                if(bottoms[0]->c==bottoms[i]->c && bottoms[0]->w==bottoms[i]->w)
                    height+=bottoms[i]->h;
                else {
                    printf("The cancat size is not correct!!\n");
                    return -1;
                }

            }
            tops[0]->setSize(height,bottoms[0]->w,bottoms[0]->c);
            return 0;
        }

        if(axis==2)
        {
            int width=bottoms[0]->w;
            for (int i = 1; i < number; i++)
            {
                if(bottoms[0]->c==bottoms[i]->c && bottoms[0]->h==bottoms[i]->h)
                    width+=bottoms[i]->w;
                else {
                    printf("The cancat size is not correct!!\n");
                    return -1;
                }

            }
            tops[0]->setSize(bottoms[0]->h,width,bottoms[0]->c);
            return 0;
        }
    }

    int Concat::forward() const
    {
        return 0;
    }
}
