

#ifndef FASTNN_PARAMDICT_H
#define FASTNN_PARAMDICT_H

#include <stdio.h>

// at most 20 parameters
#define MAX_PARAM_COUNT 20

namespace fastnn {

class Net;
class ParamDict
{
public:
    // empty
    ParamDict();
    ~ParamDict();

    // get int
    int get(int id, int def) const;
    // get float
    float get(int id, float def) const;
    // get array
    float* get(int id, float* def) const;

    // set int
    void set(int id, int i);
    // set float
    void set(int id, float f);
    // set array
    void set(int id, float* v);

protected:
    friend class Net;

    void clear();

    int load_param(FILE* fp);

protected:
    struct
    {
        int loaded;
        union { int i; float f; };
        float* v;
    } params[MAX_PARAM_COUNT];
};

} // namespace fastnn

#endif
