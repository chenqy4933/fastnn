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

    Blob& Blob::operator=( Blob& blob)
    {
        if (this == &blob)
            return *this;

        name=blob.name;
        width=blob.width;
        height=blob.height;
        channel=blob.channel;
        padChanel=blob.padChanel;
        data=blob.data;
        owner=blob.owner;
        return *this;
    }

    int inline Blob::set_ower()
    {
        owner=true;
    }

    Blob::Blob(std::string name)
    {
        name=name;
    }

    Blob::Blob(int h,int w,int c,std::string name)
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

    Blob::Blob(int h,int w,int c,std::string name,float* data_in)
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

    int Blob::create(int h,int w,int c,std::string name)
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

    int Blob::setSize( std::vector<int> size)
    {
        if(size.size() !=4)
        {
            channel = size[2];
            width = size[1];
            height = size[0];
            if (channel > 4)
                padChanel = ROUNDUP4(channel);
            else
                padChanel = channel;

            this->size = height * width * padChanel;
            return 0;
        }
        return -1;
    }

    int Blob::setSize(int *size)
    {
        channel=size[2];
        width=size[1];
        height=size[0];
        if(channel>4)
            padChanel=ROUNDUP4(channel);
        else
            padChanel=channel;

        this->size=height*width*padChanel;
        return 0;
    }

    int Blob::get_blob_shape(int *ptr)
    {
        if(ptr!=NULL)
        {
            ptr[0]=1;
            ptr[1]=height;
            ptr[2]=width;
            ptr[3]=channel;

            if(channel>4)
                padChanel=ROUNDUP4(channel);
            else
                padChanel=channel;
            return 0;
        }
        else
        {
            return -1;
        }
    }

} // namespace fastnn
