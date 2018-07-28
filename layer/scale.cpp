

#include "scale.h"

namespace fastnn {


Scale::Scale()
{
    support_inplace = true;
}

Scale::~Scale()
{
    if (scale_data!=NULL) {
        fastnn_free(scale_data);
    }
    if (bias_data!=NULL) {
        fastnn_free(bias_data);
    }
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size != -233)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data==NULL)
            return -100;
    }

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data==NULL)
            return -100;
    }

    return 0;
}


int Scale::forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const
{

    return 0;
}

} // namespace fastnn
