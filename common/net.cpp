

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
    if(conmom_ptr!=NULL)
    {
        fastnn_free(conmom_ptr);
    }
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
    //remove the deleted layer
    int current=0;
    for(int i=0;i<allLayer.size();i++) {
        Layer *thisLayer = allLayer[i];
        if (thisLayer != NULL)                 //if the layer is NULL ,this layer is fused
        {
            allLayer[current]=thisLayer;
            current++;
        }
    }
    allLayer.resize(current);
    std::vector<Layer*> serialLayer;
    //Generate the graph
    for(int i=0;i<allLayer.size();i++)
    {
        Layer * thisLayer=allLayer[i];

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
    //find out the input blob
    for(int i=0;i<allLayer.size();i++)
    {
        Layer * thisLayer=allLayer[i];

        if(thisLayer->bottoms.size()==0)
        {
            for(int j=0;j<thisLayer->tops.size();j++)
            {
                Blob * top_blob=thisLayer->tops[j];
                input[top_blob->name]=top_blob;
            }
        }
    }
    //find out the out blob
    for(int i=0;i<allLayer.size();i++)
    {
        Layer * thisLayer=allLayer[i];
        if(thisLayer->nexts.size()==0)
        {
            for(int j=0;j<thisLayer->tops.size();j++)
            {
                Blob * top_blob=thisLayer->tops[j];
                output[top_blob->name]=top_blob;
            }
        }
    }
    std::set<std::string> OKblob;
    std::stack<Layer*> backLayer;
    std::set<std::string> isExcuted;
    for(auto iter=input.begin();iter!=input.end();iter++)
    {
        OKblob.insert(iter->first);
        backLayer.push(iter->second->producer); //push the input layer to the stack
    }
    //conduct the DFS
    while(!backLayer.empty())
    {
        Layer* thisLayer=backLayer.top();
        //make sure the layer can process
        {
            bool excutable = true;
            for (auto iter = thisLayer->bottoms.begin(); iter != thisLayer->bottoms.end(); iter++) {
                if (OKblob.find((*iter)->name) == OKblob.end());
                {
                    excutable = false;
                    break;
                }
            }
            if (!excutable) {
                backLayer.pop();
                break;
            }
        }
        //excute the layer
        {
            serialLayer.push_back(thisLayer);
            for (int i = 0; i < thisLayer->tops.size(); i++)  //push all the top to the set
            {
                OKblob.insert(thisLayer->tops[i]->name);
            }
            isExcuted.insert(thisLayer->name);       //indict the layer is executed
            backLayer.pop();
        }
        //push the next to the stack
        for(auto iter=thisLayer->nexts.begin();iter!=thisLayer->nexts.end();iter++)
        {
            if(isExcuted.find((*iter)->name)==isExcuted.end())
            {
                backLayer.push(*iter);
            }
        }
    }
    //delete the no use layer
    for(int i=0;i<allLayer.size();i++)
    {
        Layer* thisLayer=allLayer[i];
        bool noUse=true;
        for(int j=0;j<serialLayer.size();j++)
        {
            if(thisLayer->name==serialLayer[j]->name)
            {
                noUse=false;
                break;
            }
        }
        if(noUse)
        {
            delete(thisLayer);
        }
    }
    allLayer=serialLayer;
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
int Net::InitNet(void)
{
    //first conduct net_optimize
    int ret=0;
    ret=organize_net();
    if(ret!=0)
    {
        printf("organize_net first wrong\n");
        return -1;
    }
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
        printf("organize_net second wrong\n");
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
    // if the number is not equal the input size ,use default size
    if(number!=input.size())
    {
        for(auto iter = input.begin(); iter != input.end(); iter++)
        {
            Blob * input_blob=iter->second;
            Layer* input_layer=input_blob->producer;
            std::vector<int> shape(4);
            shape[0]=1;
            shape[1]= static_cast<Input*>(input_layer)->h;
            shape[2]= static_cast<Input*>(input_layer)->w;
            shape[3]= static_cast<Input*>(input_layer)->c;
            input_blob->setSize(shape);
        }
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
    // infershape return the temp memory it used
    int max_temp_size=0;
    for(int i=0;i<allLayer.size();i++)
    {
        int temp_size=allLayer[i]->infershape();
        if(temp_size>max_temp_size)
            max_temp_size=temp_size;
    }
    //allocate the temp memory
    conmom_ptr=fastnn_alloc(max_temp_size);
    comom_size=max_temp_size;

    int numberOfblob=allBlob.size();
    if(numberOfblob<=0)
    {
        printf("Net is not load successfully\n");
        return -1;
    }
    std::map<std::string,int> blobReference;
    //Init all the reference to 0
    for(auto iter = allBlob.begin(); iter != allBlob.end(); iter++)
    {
        blobReference[iter->first]=0;
        blob2ptr[iter->first]=-100;
    }

    for(auto iter=input.begin();iter!=input.end();iter++)
    {
        blobReference[iter->first]=-1;     //if blobReference[i]==-1 ,it ptr is provide by user
        blob2ptr[iter->first]=-1;
    }
//    for(auto iter=output.begin();iter!=output.end();iter++)
//    {
//        blobReference[iter->first]=-1;    //if blobReference[i]==-1 ,it ptr is provide by user
//        blob2ptr[iter->first]=-1;
//    }

    //the input and output provided by the user
    //Get the reference of all the blob

    for(int i=0;i<allLayer.size();i++)
    {
        Layer* thisLayer=allLayer[i];
        for(int j=0;j<thisLayer->bottoms.size();j++)
        {
            Blob* thisBlob=thisLayer->bottoms[j];
            blobReference[thisBlob->name]+=1;
        }
    }

    //check weather all the blob is referenced
    for(auto iter=blobReference.begin();iter!=blobReference.end();iter++)
    {
        if(iter->second==0)
        {
            printf("ERRO!!! The %s blob is no referenced\n",iter->first.c_str());
            return -1;
        }
    }

    //crate the free blob pool
    std::multimap<int,int> freepool;   // memory map from size--->index_ptr
    std::vector<int> ptr_size_all;
    std::map<std::string,int> blob2ptr_all;
    int index_ptr=0;
    if(organized)
    {
        int min_size=0;
        float min_scale=0;
        for(float time=0;time<6;time+=1)   //选择最优的一种内存分配方案
        {
            freepool.clear();
            index_ptr=0;
            float scale=0.5f+time*0.1f;
            std::map<std::string,int> blob2ptr_temp=blob2ptr;
            std::vector<int> ptr_size;
            for (int i = 0; i < allLayer.size(); i++)
            {
                //
                Layer* thisLayer=allLayer[i];

                for(int j=0;j<thisLayer->tops.size();j++)
                {
                    //1. find the memory for the tops
                    if(blob2ptr_temp[thisLayer->tops[j]->name]!=-1)  //not the input or output blob
                    {
                        if(thisLayer->support_inplace)
                        {
                            blob2ptr_temp[thisLayer->tops[j]->name]=blob2ptr_temp[thisLayer->bottoms[j]->name];
                        }
                        else  //get the blobptr from the freepool, if can't find then new it
                        {
                            Blob* top_blob=thisLayer->tops[j];
                            int blob_size=top_blob->size;
                            auto iter=freepool.lower_bound(blob_size);  //choose from biger
                            if(iter!=freepool.end() && iter->first<=(int)(blob_size/scale))
                            {
                                blob2ptr_temp[top_blob->name]=iter->second;
                                ptr_size[iter->second]=std::max(blob_size,iter->first);
                                freepool.erase(iter);
                                continue;
                            }
                            iter=freepool.lower_bound((int)(blob_size*scale)); //choose from smaller
                            if(iter!=freepool.end())
                            {
                                blob2ptr_temp[top_blob->name]=iter->second;
                                ptr_size[iter->second]=std::max(blob_size,iter->first);
                                freepool.erase(iter);
                                continue;
                            }
                            ptr_size.push_back(blob_size);     // create the memory block
                            blob2ptr_temp[top_blob->name]=index_ptr;
                            index_ptr++;
                        }
                    }
                }
                //2. collect the bottom blob to the freepool
                if(!thisLayer->support_inplace)   // if layer support inplace then no need recircle ,because the memory is using.
                {
                    for(int j=0;j<thisLayer->bottoms.size();j++)
                    {
                        int bottom_index=blob2ptr_temp[thisLayer->bottoms[j]->name];
                        if(bottom_index!=-1)  //not the input or output blob
                        {
                            bool isInPool=false;
                            for(auto iter=freepool.begin();iter!=freepool.end();iter++)
                            {
                                if(iter->second==bottom_index)
                                {
                                    isInPool=true;
                                    break;
                                }
                            }
                            if(!isInPool)   // if not in pool
                            {
                                int size=ptr_size[bottom_index];
                                freepool.insert(std::make_pair(size,bottom_index));
                            }
                        }
                    }
                }
            }
            int all_size=0;
            for(int i=0;i<ptr_size.size();i++)
            {
                all_size+=ptr_size[i];
            }
            if(all_size<min_size)
            {
                min_size=all_size;
                min_scale=scale;
                blob2ptr_all=blob2ptr_temp;
                ptr_size_all=ptr_size;
            }
        }
        int index=(min_scale-0.5f)*10.f;
        blob2ptr=blob2ptr_all;
        ptrSize=ptr_size_all;
    }
    else
    {
        printf("Please organize the Net ahead\n");
        return -1;
    }
    return 0;
}

int Net::memory_alloc(void)
{
    //1. alloc all the memory and store it in the allPtr
    if(blob2ptr.size()>0 && ptrSize.size()>0)   // alloc all the memory of blob
    {
        int nNumber_ptr=ptrSize.size();
        allPtr.resize(nNumber_ptr);
        for(int i=0;i<nNumber_ptr;i++)
        {
            float* ptr=fastnn_alloc(ptrSize[i]);
            allPtr[i]=ptr;
        }
    }
    else
    {
        printf("Memory plan is not right\n");
        return -1;
    }
    //2. assign the ptr to all the blob
    for(auto iter=allBlob.begin();iter!=allBlob.end();iter++)
    {
        int ptr_index=blob2ptr[iter->first];
        if(allPtr.size()>ptr_index)
        {
            iter->second.setData(allPtr[ptr_index]);
        }
        else
        {
            printf("Memory plan is erro!!\n");
        }
    }
    return 0;
}

int Net::Forward()
{
    for(int i=0;i<allLayer.size();i++)
    {
//         allLayer[i]->forward();
    }
}


NetEngine::NetEngine(const char *param_path, const char *model_path)
{
    net=new Net();
    if(NULL==net)
    {
        printf("Net is not created!\n");
        return;
    }
    else
    {
        if(net->load_param(param_path)!=0)
        {
            printf("Net load_param error\n");
            net->clear();
            return;
        }
        if(net->load_model(model_path)!=0)
        {
            printf("Net load_param error\n");
            net->clear();
            return;
        }
    }
}

int NetEngine::set_input_shape(int *ptr, const char *name)
{
    if(ptr!=NULL && name!=NULL && net!=NULL)
    {
        if(net->input.find(name)!=net->input.end())
        {
            std::vector<int> shape(4);
            shape[0]=ptr[0];
            shape[1]=ptr[1];
            shape[2]=ptr[2];
            shape[3]=ptr[3];
            inputsize[name]=shape;
        }
        else
        {
            printf("Net dose not have the input blob %s\n",name);
            return -1;
        }
    }
    return 0;
}

int NetEngine::init()
{
    if(net!=NULL)
    {
        inited=true;
        int ret=0;
        //0. Organize the Net
        ret=net->organize_net();
        if(ret!=0)
        {
            printf("organize_net error\n");
            net->clear();
            return -1;
        }
        //1. set the input of the net;
        ret=net->set_input_size(inputsize);
        if(ret!=0)
        {
            printf("set_input_size error\n");
            net->clear();
            return -1;
        }
        //2. init the Net
        ret=net->InitNet();
        if(ret!=0)
        {
            printf("InitNet error\n");
            net->clear();
            return -1;
        }
    }
    else
    {
        printf("Net is not created!\n");
        return -1;
    }
    return 0;

}

int NetEngine::get_input_shape(int *ptr, const char *name)
{
    if(ptr!=NULL && name!=NULL && net!=NULL)
    {
        if(net->input.find(name)!=net->input.end())
        {
            Blob *input_blob = net->input[name];
            Layer *input_layer = input_blob->producer;
            ptr[0] = 1;
            ptr[1] = static_cast<Input *>(input_layer)->h;
            ptr[2] = static_cast<Input *>(input_layer)->w;
            ptr[3] = static_cast<Input *>(input_layer)->c;
            return 0;
        }
        else
        {
            printf("Net dose not have the input blob %s\n",name);
            return -1;
        }
    }
    return -1;
}

int NetEngine::forward()
{
    if(net!=NULL)
    {
        if(net->Forward())
        {
            printf("Forward error %s\n");
            return -1;
        }
    }
    return 0;
}

int NetEngine::input(const char* blob_name, Blob& in_blob)
{
    
}

// the return blob data must to be processed before the net is delete
int NetEngine::output(const char* blob_name, Blob& out_blob)
{
    if(net!=NULL)
    {
        if (blob_name != NULL)
        {
            if (net->output.find(blob_name) != net->output.end())
            {
                out_blob=*(net->output[blob_name]);

                return 0;
            }
            else
            {
                printf("Net dose not have the output blob %s\n",blob_name);
                return -1;
            }
        }
        else
        {
            printf("blob name==NULL \n");
            return -1;
        }
    }
    else
    {
        printf("Net is not created!\n");
        return -1;
    }
    return -1;
}

NetEngine::~NetEngine()
{
    if(net!=NULL)
    {
        delete(net);
    }
}

} // namespace ncnn
