

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
    register_layer_creators();
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
            if(it==allBlob.end())
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
/************************************
 *
 * @return
 */
int Net::clear()
{
    for(int i=0;i<allLayer.size();i++)
    {
        if(allLayer[i]!=NULL)
        {
            delete allLayer[i];
        }
    }
    for(int i=0;i<allPtr.size();i++)
    {
        if(allPtr[i]!=NULL)
        {
            fastnn_free(allPtr[i]);
        }
    }
    return 0;
}

/******************************
 *
 */

int Net::organize_net(void)
{
    //Generate the grap
    for(int i=0;i<allLayer.size();i++)
    {
        Layer * thisLayer=allLayer[i];
        if(thisLayer!=NULL)                 //if the layer is NULL ,this layer is fused
        {
            thisLayer->nexts.clear();
            for (int j = 0; j < thisLayer->tops.size(); j++)
            {
                Blob *top_blob = thisLayer->tops[j];
                for (int k = 0; k < top_blob->consumer.size(); k++)
                {
                    thisLayer->nexts.push_back(top_blob->consumer[k]);
                }
            }
        }
    }
    //find out the input blob
    for(int i=0;i<allLayer.size();i++)
    {
        Layer * thisLayer=allLayer[i];
        if(thisLayer!=NULL)                 //if the layer is NULL ,this layer is fused
        {
            if(thisLayer->bottoms.size()==0)
            {
                for(int j=0;j<thisLayer->tops.size();j++)
                {
                    Blob * top_blob=thisLayer->tops[j];
                    input[top_blob->name]=top_blob;
                }
            }
        }
    }
    //find out the out blob
    for(int i=0;i<allLayer.size();i++)
    {
        Layer * thisLayer=allLayer[i];
        if(thisLayer!=NULL)                 //if the layer is NULL ,this layer is fused
        {
            if(thisLayer->nexts.size()==0)
            {
                for(int j=0;j<thisLayer->tops.size();j++)
                {
                    Blob * top_blob=thisLayer->tops[j];
                    output[top_blob->name]=top_blob;
                }
            }
        }
    }
    organized=true;
    return 0;
}
/******************************
 *
 */
int Net::net_optimize()
{
    if(organized)
    {
        for(int i=0;i<allLayer.size();i++)
        {
            Layer* thisLayer=allLayer[i];
            Layer* next=thisLayer->nexts[0];
            bool fuse=(thisLayer->type==std::string("Converlution"));
            fuse =(fuse && (next->type==std::string("Scale") ||
                    next->type==std::string("BatchNormal")||next->type==std::string("Relu")));
            if(fuse)
            {
                if(!(thisLayer->bottoms.size()==1)&&(thisLayer->tops.size()==1))
                {
                    printf("Net is wrong!!!\n");
                    return -100;
                }
                fuse_layer(thisLayer,next);
                thisLayer->updata_weight(next);
            }
        }
    }
}
/******************************
 *
 */
int Net::fuse_layer(Layer* baseLayer,Layer * next)
{
    if(baseLayer!=NULL && next!=NULL)
    {
        if(baseLayer->tops[0]->consumer[0]!=next)
        {
            printf("Fuse_layer wrong!!!\n");
            return -1;
        }
        next->tops[0]->producer=baseLayer;
        baseLayer->tops[0]=next->tops[0];

        auto it=allBlob.find(next->bottoms[0]->name);
        allBlob.erase(it);
        delete next->bottoms[0];                    //delete the fuse blob

        for(int i=0;i<allLayer.size();i++)
        {
            if(allLayer[i]!=NULL && allLayer[i]==next)
            {
                allLayer[i]=NULL;
            }
        }
        delete next;
    }
    return 0;
}
/******************************
 *
 */
int Net::before_Forward(void)
{
    //first conduct net_optimize
    int ret=0;
    ret=net_optimize();
    if(ret!=0)
    {
        printf("net_optimize wrong\n");
        return -1;
    }
    //second conduct organize_net
    ret=organize_net();
    if(ret!=0)
    {
        printf("organize_net wrong\n");
        return -1;
    }
    //third conduct memory_plan
    ret=net_memory_plan();
    if(ret!=0)
    {
        printf("net_memory_plan wrong\n");
        return -1;
    }
    //forth conduct memory_alloc
    ret=memory_alloc();
    if(ret!=0)
    {
        printf("memory_alloc wrong\n");
        return -1;
    }
    return 0;

}

int Net::set_input_size(std::map<std::string,std::vector<int>> sizeOfinput)
{
    int number=sizeOfinput.size();
    if(number!=input.size())
    {
        printf("Input size in not right\n");
        return -1;
    }
    else
    {
        for(auto iter = input.begin(); iter != input.end(); iter++)
        {
            Blob * input_blob=iter->second;
            std::vector<int> shape=sizeOfinput[iter->first];
            input_blob->setSize(shape);
        }
    }
    return 0;
}

int Net::net_memory_plan()
{

    //Set the input blob size
    //infershape of all the blob include pad
    for(int i=0;i<allLayer.size();i++)
    {
        Layer* thisLayer=allLayer[i];
        thisLayer->infershape();
    }

    int numberOfblob=allBlob.size();
    if(numberOfblob<=0)
    {
        printf("Net is not load successfully\n");
        return -1;
    }
    std::map<std::string,int> blob_reference;
    //Init all the reference to 0
    for(auto iter = allBlob.begin(); iter != allBlob.end(); iter++)
    {
        blob_reference[iter->first]=0;
    }
    //Get the reference of all the blob
    for(int i=0;i<allLayer.size();i++)
    {
        Layer* thisLayer=allLayer[i];
        for(int j=0;j<thisLayer->bottoms.size();j++)
        {
            Blob* thisBlob=thisLayer->bottoms[j];
            blob_reference[thisBlob->name]+=1;
        }
    }
    for(int i=0;i<allLayer.size();i++)
    {
        Layer* thisLayer=allLayer[i];
        for(int j=0;j<thisLayer->tops.size();j++)
        {
            Blob* thisBlob=thisLayer->tops[j];
            blob_reference[thisBlob->name]-=1;
        }
    }
}

} // namespace ncnn
