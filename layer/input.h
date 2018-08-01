

#ifndef LAYER_INPUT_H
#define LAYER_INPUT_H

#include "layer.h"

namespace fastnn {

class Input : public Layer
{

public:
    Input();
    ~Input(){};

    virtual int load_param(const ParamDict& pd) override;

    virtual int infershape() override;

    int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const override;

public:
    int w;
    int h;
    int c;
};

Layer *GetInputLayer()
{
    return (Layer *)new Input();
}

} // namespace fastnn

#endif // LAYER_INPUT_H
