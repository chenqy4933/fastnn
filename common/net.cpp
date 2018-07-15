

#include "net.h"
#include "layer_register.h"
#include "modelbin.h"
#include "paramdict.h"
#include "memoryAlloc.h"

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
//    allBlob.resize(blob_count);

    ParamDict pd;

    int layer_index = 0;
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
            auto it=allBlob.find(bottom_name);
            if(it!=allBlob.end())
            //int bottom_blob_index = find_blob_index_by_name(bottom_name);
            //if (bottom_blob_index == -1)
            {
                printf("The model's layer is out of order!");
                return -1;
            }
            else
            {
                Blob temp_blob=it->second;
                temp_blob.consumer.push_back(layer);
                layer->bottoms[i]=&temp_blob;
            }
        }

        layer->tops.resize(top_count);
        for (int i=0; i<top_count; i++)
        {

            char blob_name[257];
            nscan = fscanf(fp, "%256s", blob_name);
            if (nscan != 1)
            {
                continue;
            }

            std::string name = std::string(blob_name);
            Blob top_blob(name);
//             fprintf(stderr, "new blob %s\n", blob_name);

            top_blob.producer = layer;
            allBlob[name]=top_blob;

            layer->tops[i] = &top_blob;
        }

        // layer specific params
        int pdlr = pd.load_param(fp);
        if (pdlr != 0)
        {
            printf("Param read failed of layer %s\n",layer->name.c_str());
            continue;
        }

        int lr = layer->load_param(pd);
        if (lr != 0)
        {
            printf("layer %s load param erro!\n",layer->name.c_str());
            continue;
        }

        allLayer[layer_index] = layer;

        layer_index++;
    }
    return 0;
}

int Net::load_param(const char* protopath)
{
    FILE* fp = fopen(protopath, "rb");
    if (!fp)
    {
        printf("open parma %s failed\n", protopath);
        return -1;
    }

    int ret = load_param(fp);

    fclose(fp);

    return ret;
}

int Net::load_model(FILE* fp)
{
    if (allLayer.empty())
    {
        fprintf(stderr, "network graph not ready\n");
        return -1;
    }

    // load file
    int ret = 0;

    ModelBinFromStdio mb(fp);
    for (size_t i=0; i<allLayer.size(); i++)
    {
        Layer* layer = allLayer[i];

        int lret = layer->load_model(mb);
        if (lret != 0)
        {
            printf("Load_model of layer %s failed\n", layer->name.c_str());
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
        printf("open %s erro!!\n", modelpath);
        return -1;
    }

    int ret = load_model(fp);

    fclose(fp);

    return ret;
}

int Net::clear()
{
    for(int i=0;i<allLayer.size();i++)
    {
        if(allLayer[i]!=NULL)
        {
            delete allLayer[i];
        }
    }
    return 0;
}

int Net::organize_net(void)
{
    return 0;
}

} // namespace ncnn
