//
// Created by 陈其友 on 2018/7/24.
//

#ifndef FASTNN_CONVOLUTION_H
#define FASTNN_CONVOLUTION_H

#include "layer.h"
#include "nnfunc/general_conv.h"

namespace fastnn {

class Convolution: public Layer {

public:
    ~Convolution(){
        if(weight_data!=NULL)
        {
            fastnn_free(weight_data);
            weight_data=NULL;
        }
        if(bias_data!=NULL)
        {
            fastnn_free(bias_data);
            bias_data=NULL;
        }
        if(gconv!=NULL)
        {
            fastnn_free(gconv);
            gconv=NULL;
        }
    };
    Convolution(){};

    virtual int load_param(const ParamDict& pd) override;

    virtual int load_model(const ModelBin& mb) override;

    virtual int infershape() override;

    virtual int init() override;

    int forward() const override;

public:
    int num_output;
    int kernel_w;
    int kernel_h;
    int dilation_w;
    int dilation_h;
    int stride_w;
    int stride_h;
    int pad_w;
    int pad_h;
    int bias_term;

    int weight_data_size;

    // model
    float* weight_data=NULL;
    float* bias_data=NULL;

    //func
    General_conv* gconv=NULL;
};

Layer *GetConvolutionLayer()
{
    return (Layer *) new Convolution();
}

}
#endif //FASTNN_CONVOLUTION_H
