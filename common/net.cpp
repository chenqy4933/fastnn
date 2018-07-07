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

#include "net.h"
#include "layer_register.h"
#include "modelbin.h"
#include "paramdict.h"

#include <stdio.h>
#include <string.h>


namespace fastnn {

Net::Net()
{
}

Net::~Net()
{
    clear();
}

int Net::load_param(FILE* fp)
{
    int magic = 0;
    fscanf(fp, "%d", &magic);
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    // parse
    int layer_count = 0;
    int blob_count = 0;
    fscanf(fp, "%d %d", &layer_count, &blob_count);

    allLayer.resize(layer_count);
    allBlob.resize(blob_count);

    ParamDict pd;

    int layer_index = 0;
    int blob_index = 0;
    while (!feof(fp))
    {
        int nscan = 0;

        char layer_type[257];
        char layer_name[257];
        int bottom_count = 0;
        int top_count = 0;
        nscan = fscanf(fp, "%256s %256s %d %d", layer_type, layer_name, &bottom_count, &top_count);
        if (nscan != 4)
        {
            continue;
        }
        Layer* layer = create_layer(layer_type);
        if (!layer)
        {
            printf("Create layer %s failed!!!\n",layer_type);
            clear();
            return -1;
        }

        layer->type = std::string(layer_type);
        layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d %s\n", layer_index, layer_name);

        layer->bottoms.resize(bottom_count);
        for (int i=0; i<bottom_count; i++)
        {
            char bottom_name[257];
            nscan = fscanf(fp, "%256s", bottom_name);
            if (nscan != 1)
            {
                continue;
            }

            int bottom_blob_index = find_blob_index_by_name(bottom_name);
            if (bottom_blob_index == -1)
            {
                Blob& blob = blobs[blob_index];

                bottom_blob_index = blob_index;

                blob.name = std::string(bottom_name);
//                 fprintf(stderr, "new blob %s\n", bottom_name);

                blob_index++;
            }

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(layer_index);

            layer->bottoms[i] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int i=0; i<top_count; i++)
        {
            Blob& blob = blobs[blob_index];

            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

            blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = layer_index;

            layer->tops[i] = blob_index;

            blob_index++;
        }

        // layer specific params
        int pdlr = pd.load_param(fp);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[layer_index] = layer;

        layer_index++;
    }

    return 0;
}

int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    int ret = load_param(fp);

    fclose(fp);

    return ret;
}

int Net::load_param_bin(FILE* fp)
{
    int magic = 0;
    fread(&magic, sizeof(int), 1, fp);
    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return -1;
    }

    int layer_count = 0;
    fread(&layer_count, sizeof(int), 1, fp);

    int blob_count = 0;
    fread(&blob_count, sizeof(int), 1, fp);

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex;
        fread(&typeindex, sizeof(int), 1, fp);

        int bottom_count;
        fread(&bottom_count, sizeof(int), 1, fp);

        int top_count;
        fread(&top_count, sizeof(int), 1, fp);

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %d not exists or registered\n", typeindex);
            clear();
            return -1;
        }

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index;
            fread(&bottom_blob_index, sizeof(int), 1, fp);

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index;
            fread(&top_blob_index, sizeof(int), 1, fp);

            Blob& blob = blobs[top_blob_index];

//             blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param_bin(fp);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

    return 0;
}

int Net::load_param_bin(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", protopath);
        return -1;
    }

    int ret = load_param_bin(fp);

    fclose(fp);

    return ret;
}

int Net::load_model(FILE* fp)
{
    if (layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return -1;
    }

    // load file
    int ret = 0;

    ModelBinFromStdio mb(fp);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model %d failed\n", (int)i);
            ret = -1;
            break;
        }
    }

    return ret;
}

int Net::load_model(const char* modelpath)
{
    FILE* fp = fopen(modelpath, "rb");
    if (!fp)
    {
        fprintf(stderr, "fopen %s failed\n", modelpath);
        return -1;
    }

    int ret = load_model(fp);

    fclose(fp);

    return ret;
}

int Net::load_param(const unsigned char* _mem)
{
    if ((unsigned long)_mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", _mem);
        return 0;
    }

    const unsigned char* mem = _mem;

    int magic = *(int*)(mem);
    mem += 4;

    if (magic != 7767517)
    {
        fprintf(stderr, "param is too old, please regenerate\n");
        return 0;
    }

    int layer_count = *(int*)(mem);
    mem += 4;

    int blob_count = *(int*)(mem);
    mem += 4;

    layers.resize(layer_count);
    blobs.resize(blob_count);

    ParamDict pd;

    for (int i=0; i<layer_count; i++)
    {
        int typeindex = *(int*)mem;
        mem += 4;

        int bottom_count = *(int*)mem;
        mem += 4;

        int top_count = *(int*)mem;
        mem += 4;

        Layer* layer = create_layer(typeindex);
        if (!layer)
        {
            int custom_index = typeindex & ~LayerType::CustomBit;
            layer = create_custom_layer(custom_index);
        }
        if (!layer)
        {
            fprintf(stderr, "layer %d not exists or registered\n", typeindex);
            clear();
            return 0;
        }

//         layer->type = std::string(layer_type);
//         layer->name = std::string(layer_name);
//         fprintf(stderr, "new layer %d\n", typeindex);

        layer->bottoms.resize(bottom_count);
        for (int j=0; j<bottom_count; j++)
        {
            int bottom_blob_index = *(int*)mem;
            mem += 4;

            Blob& blob = blobs[bottom_blob_index];

            blob.consumers.push_back(i);

            layer->bottoms[j] = bottom_blob_index;
        }

        layer->tops.resize(top_count);
        for (int j=0; j<top_count; j++)
        {
            int top_blob_index = *(int*)mem;
            mem += 4;

            Blob& blob = blobs[top_blob_index];

//             blob.name = std::string(blob_name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            blob.producer = i;

            layer->tops[j] = top_blob_index;
        }

        // layer specific params
        int pdlr = pd.load_param(mem);
        if (pdlr != 0)
        {
            fprintf(stderr, "ParamDict load_param failed\n");
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            fprintf(stderr, "layer load_param failed\n");
            continue;
        }

        layers[i] = layer;
    }

    return mem - _mem;
}

int Net::load_model(const unsigned char* _mem)
{
    if (layers.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return 0;
    }

    if ((unsigned long)_mem & 0x3)
    {
        // reject unaligned memory
        fprintf(stderr, "memory not 32-bit aligned at %p\n", _mem);
        return 0;
    }

    const unsigned char* mem = _mem;
    ModelBinFromMemory mb(mem);
    for (size_t i=0; i<layers.size(); i++)
    {
        Layer* layer = layers[i];

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            fprintf(stderr, "layer load_model failed\n");
            return -1;
        }
    }

    return mem - _mem;
}


} // namespace ncnn
