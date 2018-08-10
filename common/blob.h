
#ifndef FASTNN_BLOB_H
#define FASTNN_BLOB_H

#include <string>
#include <vector>
#include "layer.h"

namespace fastnn {

class Layer;

#define ROUNDUP4(x) (((x+3)>>2)<<2)

class Blob
{
public:
    // empty
    Blob(){};

    virtual ~Blob();

    Blob(std::string name);

    Blob(int heiht,int width,int channel,std::string name);

    Blob(int heiht,int width,int channel,std::string name,float* data);

    Blob& operator=( Blob& blob);

    int create(int heiht,int width,int channel,std::string name=std::string("NULL"));

    int setData(float* data);

    int setSize(std::vector<int> size);

    int setSize(int* size);

    int setSize(int height,int width,int channel);

    int set_ower();

    int get_blob_shape(int * ptr);

    int dim();

    float* data();

    int set_padparam(int up,int down,int left,int right);

    int set_pad_zero();



public:
    int h=1;
    int w=1;
    int c=1;
    int padc=1;
    //the memory of the blob
    float* data_ptr=NULL;

    int pad_up=0;
    int pad_down=0;
    int pad_left=0;
    int pad_right=0;

    size_t size=0;  // indict the pad size when pad as input, set in the infershape of layer
    // blob name
    std::string name;
    std::vector<Layer*> consumer;
    Layer* producer=NULL;
    bool owner=false;
};

}

#endif // FASTNN_BLOB_H
