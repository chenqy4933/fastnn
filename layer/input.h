

#ifndef LAYER_INPUT_H
#define LAYER_INPUT_H

#include "layer.h"

namespace fastnn {

class Input : public Layer
{
public:
    Input();

    virtual int load_param(const ParamDict& pd);

    int forward(const std::vector<Blob> bottoms,std::vector<Blob> tops) const;

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
