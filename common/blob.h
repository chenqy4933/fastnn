
#ifndef FASTNN_BLOB_H
#define FASTNN_BLOB_H

#include <string>
#include <vector>
#include "layer.h"

namespace fastnn {

class Layer;

class Blob
{
public:
    // empty
    Blob(){};

    virtual ~Blob();

    Blob(std::string name);

    Blob(int h,int w,int c,std::string name);

    Blob(int h,int w,int c,std::string name,float* data);

    Blob& operator=(const Blob& blob);

    int create(int h,int w,int c,std::string name=std::string("NULL"));

    int setData(float* data);

    int setSize(std::vector<int> size);

    int setSize(int* size);

    int set_ower();

    int get_blob_shape(int * ptr);


public:
    int width;
    int height;
    int channel;
    int padChanel;
    //the memory of the blob
    float* data;

    size_t size;  // indict the pad size when pad as input, set in the infershape of layer
    // blob name
    std::string name;
    std::vector<Layer*> consumer;
    Layer* producer;
    bool owner=false;
};

}

#endif // FASTNN_BLOB_H
