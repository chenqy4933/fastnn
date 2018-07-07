// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "scale.h"

namespace fastnn {


Scale::Scale()
{
    support_inplace = true;
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size != -233)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data.empty())
            return -100;
    }

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Scale::forward(std::vector<Blob> bottoms,std::vector<Blob> tops) const
{
    Mat& bottom_top_blob = bottoms[0].blob_mat;

    const Mat& scale_blob = scale_data;

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #pragma omp parallel for
            for (int i=0; i<w; i++)
            {
                ptr[i] = ptr[i] * scale_blob[i] + bias_data[i];
            }
        }
        else
        {
            #pragma omp parallel for
            for (int i=0; i<w; i++)
            {
                ptr[i] *= scale_blob[i];
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        if (bias_term)
        {
            #pragma omp parallel for
            for (int i=0; i<h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];
                float bias = bias_data[i];

                for (int j=0; j<w; j++)
                {
                    ptr[j] = ptr[j] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int i=0; i<h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];

                for (int j=0; j<w; j++)
                {
                    ptr[j] *= s;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (bias_term)
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];
                float bias = bias_data[q];

                for (int i=0; i<size; i++)
                {
                    ptr[i] = ptr[i] * s + bias;
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for (int q=0; q<channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];

                for (int i=0; i<size; i++)
                {
                    ptr[i] *= s;
                }
            }
        }
    }

    return 0;
}

} // namespace fastnn
