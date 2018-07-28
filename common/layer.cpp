
#include "layer.h"

#include <string.h>

namespace fastnn {

Layer::Layer()
{
    support_inplace = false;
}

Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const
{
    return 0;
}

// child dose not need to inpliment this func for no shape change and one blob
int Layer::infershape()
{
    Blob * blob=bottoms[0];
    int shape[4];
    if(blob->get_blob_shape(shape))
    {
        int ret=tops[0]->setSize(shape);
        if(ret!=0)
        {
            printf("inferShape wrong!!\n");
            return 0;
        }
    } else{
        printf("inferShape wrong!!\n");
        return 0;
    }
    // no use temp memory
    return 0;
}

int Layer::updata_weight(Layer* netx)
{
    return 0;
}

int Layer::init()
{
    return 0;
}


} // namespace fastnn
