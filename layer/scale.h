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

#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "layer.h"

namespace fastnn {

class Scale : public Layer
{
public:
    Scale();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs= vector<Blob>()) const;

public:
    // param
    int scale_data_size;
    int bias_term;

    // model
    Mat scale_data;
    Mat bias_data;
};

}

#endif // LAYER_SCALE_H
