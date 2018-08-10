//
// Created by 陈其友 on 2018/7/24.
//

#include "convolution.h"

namespace fastnn {

    int Convolution::load_param(const fastnn::ParamDict &pd)
    {
        num_output = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        dilation_w = pd.get(2, 1);
        dilation_h = pd.get(12, dilation_w);
        stride_w = pd.get(3, 1);
        stride_h = pd.get(13, stride_w);
        pad_w = pd.get(4, 0);
        pad_h = pd.get(14, pad_w);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);

        return 0;
    }

    int Convolution::load_model(const fastnn::ModelBin &mb)
    {

        float* weight= mb.load(weight_data_size, 0);
        int inputc=weight_data_size/(kernel_h*kernel_w*num_output);
        int padc=ROUNDUP4(inputc);
        weight_data=fastnn_alloc(kernel_h*kernel_w*padc*num_output*sizeof(float));
        memset(weight_data,0.0f,kernel_h*kernel_w*padc*num_output* sizeof(float));
        //do weight transport
        for(int n=0;n<num_output;n++)
        {
            float* ptr_dst=weight_data+n*padc*kernel_h*kernel_w;
            float* ptr_src=weight+n*inputc*kernel_h*kernel_w;
            for (int i = 0; i < kernel_h * kernel_w; i++)
            {
                for (int j = 0; j < inputc; j++)
                {
                    ptr_dst[i * padc + j] = ptr_src[j*kernel_h*kernel_w+i];
                }
            }
        }
        fastnn_free(weight);
        if (weight_data==NULL)
            return -1;
        if (bias_term)
        {
            float* bias=mb.load(num_output, 1);
            if(ROUNDUP4(num_output)==num_output)
            {
                bias_data=bias;
            }
            else
            {
                bias_data=fastnn_alloc(ROUNDUP4(num_output)*sizeof(float));
                memset(bias_data,0.0f,ROUNDUP4(num_output)* sizeof(float));
                memcpy(bias_data,bias,num_output* sizeof(float));
                fastnn_free(bias);
            }

            if (bias_data==NULL)
                return -1;
        }
        return 0;
    }

    int Convolution::infershape()
    {

        Blob* bottom_blob = bottoms[0];
        Blob* top_blob = tops[0];


        if (bottom_blob->size <= 0)
        {
            printf("the bottoom blob size of %s is empty.",bottom_blob->name.c_str());
            return -1;
        }


        int w = bottom_blob->w;
        int h = bottom_blob->h;

        int nOut_w=0,nOut_h=0;


        int nPad_down=0,nPad_up=0,nPad_left=0,nPad_right=0;

        const int extented_kernel_w=dilation_w*(kernel_w-1)+1;
        const int extented_kernel_h=dilation_h*(kernel_h-1)+1;

        if (pad_w > 0 || pad_h > 0)
        {
            nPad_up=pad_h;
            nPad_down=pad_h;
            nPad_right=pad_w;
            nPad_left=pad_w;
        }
        if (pad_w == -233 && pad_h == -233)
        {
            int nAllpad_w=extented_kernel_w+(w-1)/stride_w*stride_w-w;
            int nAllpad_h=extented_kernel_h+(h-1)/stride_h*stride_h-h;

            nPad_left=nAllpad_w/2;
            nPad_right=nAllpad_w-nPad_left;
            nPad_up=nAllpad_h/2;
            nPad_down=nAllpad_h-nPad_up;
        }

        nOut_w=(w+nPad_left+nPad_right-extented_kernel_w)/stride_w+1;
        nOut_h=(h+nPad_up+nPad_down-extented_kernel_h)/stride_h+1;


        top_blob->setSize(nOut_h,nOut_w,num_output);
        top_blob->set_padparam(nPad_up,nPad_down,nPad_left,nPad_right);

        return 0;
    }

    int Convolution::init()
    {
        gconv=new General_conv();
        gconv->init(kernel_w,kernel_h,dilation_w,dilation_h,
                    stride_w,stride_h,bias_term,weight_data,bias_data);
        return 0;
    }

    int Convolution::forward() const
    {
        return 0;
    }


}
