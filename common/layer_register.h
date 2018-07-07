//
// Created by 陈其友 on 2018/7/7.
//

#ifndef FASTNN_LAYER_REGISTER_H
#define FASTNN_LAYER_REGISTER_H

#include "layer.h"
#include <string>

//////////////////////////////all support layer
#include"layer/input.h"
#include"layer/scale.h"
/////////////////

namespace fastnn
{
    class LayerRegistry
    {
    public:
        typedef Layer *(*Creator)(void);
        typedef std::map<string, Creator> CreatorRegistry;

        static CreatorRegistry &Registry()
        {
            static CreatorRegistry *g_registry_ = new CreatorRegistry();
            return *g_registry_;
        }

        // Adds a creator.
        static void AddCreator(const string &type, Creator creator)
        {
            CreatorRegistry &registry = Registry();
            registry[type] = creator;
        }

        // Get a layer using a LayerParameter.
        static Layer *CreateLayer(std::string layer_type)
        {
            CreatorRegistry &registry = Registry();
            if (registry.find(layer_type) != registry.end())
            {
                return registry[layer_type]();
            }
            else
            {
                fprintf(stderr, "fastnn does not  support the %s layer!! \n", type.c_str());
                return NULL;
            }
        }

        static vector<string> LayerTypeList()
        {
            CreatorRegistry &registry = Registry();
            vector<string> layer_types;
            for (typename CreatorRegistry::iterator iter = registry.begin();
                 iter != registry.end(); ++iter)
            {
                layer_types.push_back(iter->first);
            }
            return layer_types;
        }

    private:
        // Layer registry should never be instantiated - everything is done with its
        // static variables.
        LayerRegistry() {}
    };

    class LayerRegisterer
    {
    public:
        LayerRegisterer(const string &type,
                        Layer * (*creator)(void))
        {
            LayerRegistry::AddCreator(type, creator);
        }
    };

    Layer* create_layer(std::string layer_type)
    {
        return LayerRegistry::CreateLayer(layer_type);
    }

    void register_layer_creators()
    {
        REGISTER_LAYER_CREATOR(Input, Input);
//        REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
//        REGISTER_LAYER_CREATOR(DepthwiseConvolution, GetDepthwiseConvolutionLayer);
//        REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);
//        REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
//        REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);
//        REGISTER_LAYER_CREATOR(Dropout, GetDropoutLayer);
//        REGISTER_LAYER_CREATOR(ReLU, GetReluLayer);
//        REGISTER_LAYER_CREATOR(PReLU, GetPReluLayer);
        REGISTER_LAYER_CREATOR(Scale, Scale);
//        REGISTER_LAYER_CREATOR(Slice, GetSliceLayer);
//        REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);
//        REGISTER_LAYER_CREATOR(Eltwise, GetEltwiseLayer);
//        REGISTER_LAYER_CREATOR(InnerProduct, GetInnerProductLayer);
//        REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);
//        REGISTER_LAYER_CREATOR(Filter, GetFilterLayer);
    }

#define REGISTER_LAYER_CREATOR(type, creator) \
    static LayerRegisterer g_creator_f_##type(#type, creator);

}

#endif //FASTNN_LAYER_REGISTER_H
