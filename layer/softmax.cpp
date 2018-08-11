//
// Created by 陈其友 on 2018/7/24.
//

#include "softmax.h"
#include "math.h"

namespace fastnn {

    Softmax::Softmax()
    {
        support_inplace = true;
    }

    int Softmax::load_param(const ParamDict& pd)
    {
        axis = pd.get(0, 0);

        return 0;
    }

    int Softmax::forward() const
    {
        int dim=bottoms[0]->dim();
        if(dim==1)
        {
            const uint32_t class_count = bottoms[0]->w;
            const uint32_t batch_size = class_count;


            const float *input_ptr = bottoms[0]->data();
            float *output_ptr = tops[0]->data();

            float max_val = input_ptr[0];
            for (uint32_t c = 1; c < class_count; ++c)
            {
                max_val=std::max(max_val,input_ptr[c]);
            }

            float sum = 0;
            for (uint32_t c = 0; c < class_count; ++c)
            {
                float exp_value = exp(input_ptr[c] - max_val);
                sum += exp_value;
                output_ptr[c] = exp_value;
            }

            for (uint32_t c = 0; c < class_count; ++c)
            {
                output_ptr[c] /= sum;
            }

            return 0;
        }
        else if(dim==3 && axis==0)
        {
            const uint32_t class_count = bottoms[0]->c;
            const uint32_t class_size = bottoms[0]->h * bottoms[0]->w;

            for (uint32_t k = 0; k < class_size; ++k)
            {
                const float *input_ptr = bottoms[0]->data()+k*bottoms[0]->padc;
                float *output_ptr = tops[0]->data()+k*tops[0]->padc;

                float max_val = input_ptr[0];
                uint32_t channel_offset = 0;
                for (uint32_t c = 1; c < class_count; ++c)
                {
                    max_val=std::max(max_val,input_ptr[channel_offset]);
                    channel_offset += 1;
                }

                channel_offset = 0;
                float sum = 0;
                for (uint32_t c = 0; c < class_count; ++c)
                {
                    float exp_value = exp(input_ptr[channel_offset] - max_val);
                    sum += exp_value;
                    output_ptr[channel_offset] = exp_value;
                    channel_offset += 1;
                }

                channel_offset = 0;
                for (uint32_t c = 0; c < class_count; ++c)
                {
                    output_ptr[channel_offset] /= sum;
                    channel_offset += 1;
                }
            }

            return 0;
        }

        return -1;
    }
}