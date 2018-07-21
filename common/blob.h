
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

    Blob(int w,int h,int c,std::string name);

    Blob(int w,int h,int c,std::string name,float* data);

    Blob& operator=(const Blob& blob);

    int create(int w,int h,int c,std::string name=std::string("NULL"));

    int setData(float* data);

    int setSize(std::vector<int> size);


public:
    int width;
    int height;
    int channel;
    int padChanel;
    //the memory of the blob
    float* data;

    size_t size;
    // blob name
    std::string name;
    std::vector<Layer*> consumer;
    Layer* producer;
    bool owner=false;
};

}

#endif // FASTNN_BLOB_H
