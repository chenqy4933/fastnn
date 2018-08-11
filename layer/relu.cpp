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
        int pad_left=tops[0]->pad_left;
        int pad_right=tops[0]->pad_right;
        int pad_down=tops[0]->pad_down;
        int pad_up=tops[0]->pad_up;

        float* pIn=bottoms[0]->data();
        float* pOut=tops[0]->data();

        int inw=bottoms[0]->w;
        int inh=bottoms[0]->h;
        int in_padc=bottoms[0]->padc;
        if(pad_down || pad_left || pad_right ||pad_up)
        {
            float* in_end=pIn+inw*inh*in_padc;
            float* out_end=pOut+(inw+pad_left+pad_right)*(inh+pad_up+pad_down)*in_padc;
            for(int h=0;h<inh;h++)
            {
                float* in=in_end-h*inw*in_padc;
                float* out=out_end-h*(inw+pad_left+pad_right)*in_padc+pad_left*in_padc;
                for(int w=0;w<inw*in_padc;w++)
                {
                    if(in[w]<0)
                        out[w]=slope*in[w];
                    else
                        out[w]=in[w];
                }
            }
            return 0;
        }
        else
        {
            int size=inw*inh*in_padc;
            for(int i=0;i<size;i++)
            {
                if(pIn[i]<0)
                    pOut[i]=slope*pIn[i];
                else
                    pOut[i]=pIn[i];
            }
            return 0;
        }
    }
}
