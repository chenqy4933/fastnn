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
        int number_out=tops.size();
        for(int i=0;i<number_out;i++)
        {
            bool need_pad=tops[i]->pad_down||tops[i]->pad_up||tops[i]->pad_right||tops[i]->pad_left;
            bool need_copy=false;
            for(int n=0;n<tops[i]->consumer.size();n++)
            {
                need_copy=(need_copy||tops[i]->consumer[n]->support_inplace);
            }
            if(need_copy || need_pad)
            {
                int pad_left=tops[0]->pad_left;
                int pad_right=tops[0]->pad_right;
                int pad_down=tops[0]->pad_down;
                int pad_up=tops[0]->pad_up;

                float* pIn=bottoms[0]->data();
                float* pOut=tops[0]->data();

                int inw=bottoms[0]->w;
                int inh=bottoms[0]->h;
                int inc=bottoms[0]->c;
                int in_padc=bottoms[0]->padc;

                float* in_end=pIn+inw*inh*in_padc;
                float* out_end=pOut+(inw+pad_left+pad_right)*(inh+pad_up+pad_down)*in_padc;
                for(int h=0;h<inh;h++)
                {
                    float* in=in_end-h*inw*in_padc;
                    float* out=out_end-h*(inw+pad_left+pad_right)*in_padc+pad_left*in_padc;
                    for(int w=0;w<inw*inc;w++)
                    {
                        memcpy(out,in,inw*inc*sizeof(float));
                    }
                }
                return 0;
            }
            else   //else share the memory ,so need not copy
                return 0;

        }
        return -1;
    }
}
