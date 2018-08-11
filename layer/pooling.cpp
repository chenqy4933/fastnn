//
// Created by 陈其友 on 2018/7/24.
//

#include "pooling.h"

namespace fastnn {

    Pooling::Pooling()
    {

        support_inplace = false;
    }

    int Pooling::load_param(const ParamDict& pd)
    {
        pooling_type = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        stride_w = pd.get(2, 1);
        stride_h = pd.get(12, stride_w);
        pad_left = pd.get(3, 0);
        pad_right = pd.get(14, pad_left);
        pad_top = pd.get(13, pad_left);
        pad_bottom = pd.get(15, pad_top);
        global_pooling = pd.get(4, 0);
        pad_mode = pd.get(5, 0);

        return 0;
    }

    int Pooling::infershape()
    {
        Blob* bottom_blob = bottoms[0];
        Blob* top_blob = tops[0];

        if (bottom_blob->size <= 0)
        {
            printf("the bottoom blob [%s] is empty.",bottom_blob->name.c_str());
            return -1;
        }

        int w = bottom_blob->w;
        int h = bottom_blob->h;

        if (global_pooling)
        {
            int channel=bottom_blob->c;
            top_blob->setSize(1,channel,1);
            return 0;
        } else{
            int wtailpad = 0;
            int htailpad = 0;
            int height=0;
            int width=0;
            if (pad_mode == 0) // full padding
            {
                int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
                int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

                if (wtail != 0)
                    wtailpad = stride_w - wtail;
                if (htail != 0)
                    htailpad = stride_h - htail;

                pad_right+=wtailpad;
                pad_bottom+=htailpad;

                height=w+pad_top + pad_bottom;
                width=w+pad_left + pad_right;

            }
//            else if (pad_mode == 1) // valid padding
//            {
//
//            }
            else if (pad_mode == 2) // tensorflow padding=SAME
            {
                int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
                int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;

                pad_left=wpad / 2;
                pad_right=wpad - wpad / 2;
                pad_top=hpad / 2;
                pad_bottom=hpad - hpad / 2;

            }
            height=w+pad_top + pad_bottom;
            width=w+pad_left + pad_right;

            int outw = (width - kernel_w) / stride_w + 1;
            int outh = (height - kernel_h) / stride_h + 1;

            tops[0]->set_padparam(pad_left,pad_right,pad_bottom,pad_top);

            tops[0]->setSize(outh,outw,bottoms[0]->c);

            return 0;
        }
    }

    int Pooling::forward() const
    {
        Blob* input=bottoms[0];
        Blob* output=tops[0];
        if(global_pooling)
        {
            float* out_ptr=output->data();
            float* in_ptr=input->data();
            int size=input->w*input->h;
            int channel=input->c;
            if(pooling_type==PoolMethod_MAX)
            {
                for(int c=0;c<channel;c++)
                {
                    out_ptr[c]=in_ptr[c];
                }
                for(int i=0;i<size;i++)
                {
                    float* in=in_ptr+i*input->padc;
                    for(int c=0;c<channel;c++)
                    {
                        out_ptr[c]=std::max(out_ptr[c],in[c]);
                    }
                }
            } else{
                for(int c=0;c<channel;c++)
                {
                    out_ptr[c]=0.0f;
                }
                for(int i=0;i<size;i++)
                {
                    float* in=in_ptr+i*input->padc;
                    for(int c=0;c<channel;c++)
                    {
                        out_ptr[c]+=in[c];
                    }
                }
                for(int c=0;c<channel;c++)
                {
                    out_ptr[c]/=size;
                }

            }
            return 0;
        }
        else
        {
            if(bottoms[0]->set_pad_zero()!=0)
            {
                printf("set_pad_zero error!!\n");
                return -1;
            }

            int inw=input->w+input->pad_left+input->pad_right;
            int inh=input->h+input->pad_up+input->pad_down;
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
            if(pooling_type==PoolMethod_MAX)
            {
                for(int h=0;h<outh;h++)
                {
                    float* in=pInput+h*stride_h*inw*inc;
                    float* out=pOut+h*(pad_w+outw)*pad_c;
                    for(int w=0;w<outw;w++)
                    {
                        for(int c=0;c<outc;c++)
                        {
                            float max=in[c];
                            for(int kh=0;kh<kernel_h;kh++)
                            {
                                float* iner_in=in+kh*inw*inc+c;
                                for(int kw=0;kw<kernel_w;kw++)
                                {
                                    max=std::max(iner_in[kw*inc],max);
                                }
                            }
                            out[c] = max;
                        }
                        in+=stride_w*inc;
                        out+=pad_c;
                    }
                }
            }
            else
            {
                for(int h=0;h<outh;h++)
                {
                    float* in=pInput+h*stride_h*inw*inc;
                    float* out=pOut+h*(pad_w+outw)*pad_c;
                    for(int w=0;w<outw;w++)
                    {
                        for(int c=0;c<outc;c++)
                        {
                            float sum=0.0f;
                            for(int kh=0;kh<kernel_h;kh++)
                            {
                                float* iner_in=in+kh*inw*inc+c;
                                for(int kw=0;kw<kernel_w;kw++)
                                {
                                    sum+=iner_in[kw*inc];
                                }
                            }
                            int size_w=kernel_w;
                            int size_h=kernel_h;
                            if(w-pad_left<0)
                                size_w+=(w-pad_left);
                            if(w+kernel_w+pad_right>inw)
                                size_w+=(inw-(w+kernel_w+pad_right));
                            if(h-pad_up<0)
                                size_h+=(h-pad_up);
                            if(h+kernel_h+pad_down>inh)
                                size_h+=(inh-(h+kernel_h+pad_down));
                            out[c] = sum/(size_w*size_h);
                        }
                        in+=stride_w*inc;
                        out+=pad_c;
                    }
                }
            }
            return 0;
        }
        return 0;
    }
}