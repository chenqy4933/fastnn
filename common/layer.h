
#ifndef FASTNN_LAYER_H
#define FASTNN_LAYER_H

#include <stdio.h>
#include <string>
#include <vector>
#include "blob.h"
#include "modelbin.h"
#include "paramdict.h"


namespace fastnn {

class Blob;
class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

public:
    // support inplace inference
    bool support_inplace;

public:

    virtual int infershape(const std::vector<Blob>& bottom_blobs);
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Blob>& bottom_blobs, std::vector<Blob>& top_blobs) const;

public:
    // layer type name
    std::string type;
    // layer name
    std::string name;
    // blob index which this layer needs as input
    std::vector<Blob*> bottoms;
    // blob index which this layer produces as output
    std::vector<Blob*> tops;
    // all the next layers
    std::vector<Layer*> nexts;
};

} // namespace fastnn

#endif
