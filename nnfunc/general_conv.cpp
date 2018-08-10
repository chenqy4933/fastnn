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
        bottoms[0]->set_pad_zero();
    }
}