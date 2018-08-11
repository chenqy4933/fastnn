//
// Created by 陈其友 on 2018/7/24.
//

#include "concat.h"
#include "nnfunc/general_pad.h"

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
        int input_num=bottoms.size();

        int pad_width = tops[0]->w + tops[0]->pad_right + tops[0]->pad_left;
        int pad_height = tops[0]->h + tops[0]->pad_up + tops[0]->pad_down;
        int pad_channel = tops[0]->padc;

        float * pTop=tops[0]->data()+pad_channel*(pad_width*tops[0]->pad_up+tops[0]->pad_left);
        if(axis==1)
        {
            int height=tops[0]->h;
            int width=tops[0]->w;
            for(int num=0;num<input_num;num++)
            {
                float* source=bottoms[0]->data();
                int current_c=bottoms[0]->c;
                float* dst=pTop+current_c;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        memcpy(dst,source,current_c*sizeof(float));
                        source+=current_c;
                        dst+=pad_channel;
                    }
                    dst+=(pad_width-tops[0]->w)*pad_channel;
                }
            }
            return 0;
        }
        else
            return -1;
    }
}
