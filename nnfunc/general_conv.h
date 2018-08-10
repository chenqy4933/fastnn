//
// Created by 陈其友 on 2018/8/5.
//

#ifndef FASTNN_GENERAL_CONV_H
#define FASTNN_GENERAL_CONV_H

#include "common/blob.h"

namespace fastnn {

    class General_conv {

    public:
        int init(int kernelw,int kernelh,int dilationw,int dilationh,int stridew,
                 int strideh,int biasterm,float* weightdata,float* biasdata);

        int execute(std::vector<Blob*> bottoms,std::vector<Blob*> tops);

    public:
        int num_output;
        int num_input;
        int kernel_w;
        int kernel_h;
        int dilation_w;
        int dilation_h;
        int stride_w;
        int stride_h;
        int bias_term;

        // model
        float* weight_data=NULL;
        float* bias_data=NULL;
    };
}


#endif //FASTNN_GENERAL_CONV_H
