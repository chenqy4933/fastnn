

#include "modelbin.h"
#include "memoryAlloc.h"

#include <stdio.h>
#include <string.h>
#include <vector>

namespace fastnn
{

ModelBinFromStdio::ModelBinFromStdio(FILE* _binfp) : binfp(_binfp)
{
}

float* ModelBinFromStdio::load(int w, int type) const
{
    if (!binfp)
        return NULL;

    if (type == 0)
    {
        int nread;

        union
        {
            struct
            {
                unsigned char f0;
                unsigned char f1;
                unsigned char f2;
                unsigned char f3;
            };
            unsigned int tag;
        } flag_struct;

        nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
        if (nread != 1)
        {
            printf("ModelBinFromStdio load failed %d\n", nread);
            return NULL;
        }

        unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;

        if (flag_struct.tag == 0x01306B47)
        {
            // half-precision data
            int align_data_size = alignSize(w * sizeof(unsigned short), 4);
            std::vector<unsigned short> float16_weights;
            float16_weights.resize(align_data_size);
            nread = fread(float16_weights.data(), align_data_size, 1, binfp);
            if (nread != 1)
            {
                printf("ModelBinFromStdio load float16_weights failed %d\n", nread);
                return NULL;
            }
            float* weight16=fastnn_alloc(w* sizeof(float));
            for(int i=0;i<w;i++)
            {
                weight16[i]=half2float(float16_weights[i]);
            }
            return weight16;
        }

        float* data=fastnn_alloc(w* sizeof(float));
        if (data==NULL)
            return NULL;

        if (flag != 0)
        {
            // quantized data
            float quantization_value[256];
            nread = fread(quantization_value, 256 * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                printf("ModelBinFromStdio load quantization_value failed %d\n", nread);
                fastnn_free(data);
                return NULL;
            }

            int align_weight_data_size = alignSize(w * sizeof(unsigned char), 4);
            std::vector<unsigned char> index_array;
            index_array.resize(align_weight_data_size);
            nread = fread(index_array.data(), align_weight_data_size, 1, binfp);
            if (nread != 1)
            {
                printf("ModelBinFromStdio load array failed %d\n", nread);
                fastnn_free(data);
                return NULL;
            }

            float* ptr = data;
            for (int i = 0; i < w; i++)
            {
                ptr[i] = quantization_value[ index_array[i] ];
            }
        }
        else if (flag_struct.f0 == 0)
        {
            // raw data
            nread = fread(data, w * sizeof(float), 1, binfp);
            if (nread != 1)
            {
                printf("ModelBinFromStdio load weight failed %d\n", nread);
                fastnn_free(data);
                return NULL;
            }
        }

        return data;
    }
    else if (type == 1)
    {
        float* data=fastnn_alloc(w* sizeof(float));
        if (data==NULL)
            return NULL;

        // raw data
        int nread = fread(data, w * sizeof(float), 1, binfp);
        if (nread != 1)
        {
            printf("ModelBinFromStdio load weight_data failed %d\n", nread);
            fastnn_free(data);
            return NULL;
        }

        return data;
    }
    else
    {
        printf("ModelBin load type %d not implemented\n", type);
        return NULL;
    }

    return NULL;
}





} // namespace fastnn
