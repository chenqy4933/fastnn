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

#ifndef NCNN_BLOB_H
#define NCNN_BLOB_H

#include <string>
#include <vector>
#include "mat.h"
#include "layer.h"

namespace fastnn {

class Blob
{
public:
    // empty
    Blob();

    Blob(Mat & mat);

    Blob(std::string name);

    Blob(int w,int h,int c,std::string name);

    Blob& operator=(const Blob& blob);

    int clone_mat(Blob& blob);

    int create(int w,int h,int c,std::string& name="");

    int calculte_memory();

public:
    int width;
    int height;
    int channel;
    //the memory of the blob
    Mat blob_mat;
    // blob name
    std::string name;
    Layer * consumer;
    Layer * producer;
};

}

#endif // NCNN_BLOB_H
