

#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "layer.h"

namespace fastnn {

class Scale : public Layer
{
public:
    Scale();
    ~Scale();

    virtual int load_param(const ParamDict& pd) override;

    virtual int load_model(const ModelBin& mb) override;

    int forward() const override;

public:
    // param
    int scale_data_size;
    int bias_term;

    // model
    float* scale_data=NULL;
    float* bias_data=NULL;
};

Layer *GetScaleLayer()
{
    return (Layer *)new Scale();
}

}

#endif // LAYER_SCALE_H
