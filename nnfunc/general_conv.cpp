//
// Created by 陈其友 on 2018/8/5.
//

#include "general_conv.h"

namespace fastnn{

    int General_conv::init(int kernelw,int kernelh,int dilationw,int dilationh,int stridew,
             int strideh,int biasterm,float* weightdata,float* biasdata)
    {
        kernel_w=kernelw;
        kernel_h=kernelh;
        dilation_w=dilationw;
        dilation_h=dilationh;
        stride_w=stridew;
        stride_h=strideh;
        bias_term=biasterm;
        weight_data=weightdata;
        bias_data=biasdata;
        return 0;
    }

    int General_conv::execute(std::vector<Blob*> bottoms,std::vector<Blob*> tops)
    {

        if(bottoms[0]->set_pad_zero()!=0)
        {
            printf("set_pad_zero error!!\n");
            return -1;
        }
        Blob* input=bottoms[0];
        Blob* output=tops[0];

        int inw=input->w+input->pad_left+input->pad_right;
        int inc=input->padc;

        int outw=output->w;
        int outh=output->h;
        int outc=output->c;

        int pad_left=output->pad_left;
        int pad_right=output->pad_right;
        int pad_up=output->pad_up;
        int pad_down=output->pad_down;

        int pad_c=output->padc;
        int pad_w=pad_left+pad_right;

        int kernel_size=kernel_w*kernel_h;

        float* pInput=input->data();
        float* pOut=output->data()+(pad_up*pad_w+pad_left)*pad_c;

        for(int h=0;h<outh;h++)
        {
            float* in=pInput+h*stride_h*inw*inc;
            float* out=pOut+h*(pad_w+outw)*pad_c;
            for(int w=0;w<outw;w++)
            {
                for(int c=0;c<outc;c++)
                {
                    float* kernel=weight_data+kernel_size*inc*c;
                    *out=bias_term?bias_data[c]:0.0f;
                    float sum=0.0f;
                    for(int kh=0;kh<kernel_h;kh++)
                    {
                        float* iner_in=in+kh*dilation_h*inw*inc;
                        float* iner_ker=kernel+kh*kernel_w*inc;
                        for(int kw=0;kw<kernel_w;kw++)
                        {
                            for(int cin=0;cin<inc;cin++)
                            {
                                sum+=kernel[cin]*iner_in[cin];
                            }
                            iner_in+=dilation_w*inc;
                            iner_ker+=inc;
                        }
                    }
                }
                in+=stride_w*inc;
                out+=pad_c;
            }
        }
        return 0;
    }
}