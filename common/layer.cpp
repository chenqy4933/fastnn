
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

int Layer::infershape(const std::vector<Blob>& bottom_blobs)
{
    return 0;
}

} // namespace fastnn
