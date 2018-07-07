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

#include "blob.h"

namespace fastnn {

Blob::Blob()
{
    name="no_define";
    width=0;
    height=0;
    channel=0;
}

Blob::Blob(Mat & mat)
{
    name="no_define";
    width=mat.w;
    height=mat.h;
    channel=mat.real_c;
    blob_mat=mat;
}

inline Blob& Blob::operator=(const Blob& blob)
{
    if (this == &blob)
        return *this;

    name=blob.name;
    width=blob.width;
    height=blob.height;
    channel=blob.channel;
    blob_mat=blob.mat;
    return *this;
}

int Blob::create(int w,int h,int c,std::string name)
{
    name=name;
    width=w;
    height=h;
    channel=c;
    blob_mat=Mat(w,h,c);
}

Blob::Blob(int w,int h,int c,std::string name)
{
    name=name;
    width=w;
    height=h;
    channel=c;
}

Blob::Blob(std::string name,Mat mat)
{
    if(width==mat.w && height==mat.h && channel==mat.real_c)
    {
        this->name = name;
        this->blob_mat = mat;
    }
}

int Blob::clone_mat(Blob& blob)
{
    blob_mat=blob.blob_mat.clone();
    return 0;
}

} // namespace fastnn
