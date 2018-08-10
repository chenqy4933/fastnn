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
        return 0;
    }
}