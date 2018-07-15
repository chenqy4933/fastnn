

#ifndef FASTNN_MODELBIN_H
#define FASTNN_MODELBIN_H

#include <stdio.h>

namespace fastnn {

class Net;
class ModelBin
{
public:
    // element type
    // 0 = auto
    // 1 = float32
    // 2 = float16
    // 3 = uint8
    // load vec
    virtual float* load(int w, int type) const = 0;

};

class ModelBinFromStdio : public ModelBin
{
public:
    // construct from file
    ModelBinFromStdio(FILE* binfp);

    virtual float* load(int w, int type) const;

protected:
    FILE* binfp;
};

} // namespace fastnn

#endif // FASTNN_MODELBIN_H
