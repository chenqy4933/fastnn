#include "blob.h"
#include "memoryAlloc.h"

namespace fastnn {

#define ROUNDUP4(x) (((x+3)>>2)<<2)

    inline Blob& Blob::operator=(const Blob& blob)
    {
        if (this == &blob)
            return *this;

        name=blob.name;
        width=blob.width;
        height=blob.height;
        channel=blob.channel;
        padChanel=blob.padChanel;
        return *this;
    }

    Blob::Blob(std::string name)
    {
        name=name;
    }

    Blob::Blob(int w,int h,int c,std::string name)
    {
        name=name;
        width=w;
        height=h;
        channel=c;
        if(c>4)
            padChanel=ROUNDUP4(c);
        else
            padChanel=c;

    }

    Blob::Blob(int w,int h,int c,std::string name,float* data_in)
    {
        name=name;
        width=w;
        height=h;
        channel=c;
        if(c>4)
            padChanel=ROUNDUP4(c);
        else
            padChanel=c;
        this->data=data_in;
    }

    int Blob::create(int w,int h,int c,std::string name)
    {
        name=name;
        width=w;
        height=h;
        channel=c;
        if(c>4)
            padChanel=ROUNDUP4(c);
        else
            padChanel=c;
        data=fastnn_alloc(w*h*padChanel);
    }

    int Blob::setData(Blob & blob, float* data_in)
    {
        if(data_in!=NULL)
        {
            blob.data=data_in;
        }
        else
            return -1;
        return 0;
    }

} // namespace fastnn
