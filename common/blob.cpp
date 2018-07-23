#include "blob.h"
#include "memoryAlloc.h"

namespace fastnn {

#define ROUNDUP4(x) (((x+3)>>2)<<2)

    Blob::~Blob()
    {
        if(owner==true)
        {
            fastnn_free(data);
            data=NULL;
        }
    }

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
        size=w*h*padChanel;
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
        size=w*h*padChanel;
        owner=true;
    }

    int Blob::setData(float* data_in)
    {
        if(data_in!=NULL)
        {
            this->data=data_in;
        }
        else
            return -1;
        return 0;
    }

    int Blob::setSize(std::vector<int> size)
    {
        int dim=size.size();
        if(dim!=3)
        {
            printf("set size is wrong!\n");
            return -1;
        }
        else
        {
            channel=size[0];
            height=size[1];
            width=size[2];
            if(channel>4)
                padChanel=ROUNDUP4(channel);
            else
                padChanel=channel;
        }
        this->size=height*width*padChanel;
        return 0;
    }

} // namespace fastnn
