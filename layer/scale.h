

#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "layer.h"

namespace fastnn {

class Scale : public Layer
{
public:
    ~Scale();
    Scale();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const;

public:
    // param
    int scale_data_size;
    int bias_term;

    // model
    float* scale_data;
    float* bias_data;
};

Layer *GetScaleLayer()
{
    return (Layer *)new Scale();
}

}

#endif // LAYER_SCALE_H
