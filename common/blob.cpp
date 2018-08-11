#include "blob.h"
#include "memoryAlloc.h"

namespace fastnn {


    Blob::~Blob()
    {
        if(owner==true)
        {
            fastnn_free(data_ptr);
            data_ptr=NULL;
        }
    }

    Blob& Blob::operator=( Blob& blob)
    {
        if (this == &blob)
            return *this;
        name=blob.name;
        w=blob.w;
        h=blob.h;
        c=blob.c;
        padc=blob.padc;
        data_ptr=blob.data_ptr;
        owner=blob.owner;

        size=blob.size;
        consumer=blob.consumer;
        producer=blob.producer;
        owner=blob.owner;

        pad_up=blob.pad_up;
        pad_down=blob.pad_down;
        pad_left=blob.pad_left;
        pad_right=blob.pad_right;
        return *this;
    }

    int inline Blob::set_ower()
    {
        owner=true;
        return 0;
    }

    Blob::Blob(std::string name)
    {
        this->name=name;
    }

    Blob::Blob(int height,int width,int channel,std::string name)
    {
        name=name;

        h=height;
        w=width;
        c=channel;

        if(c>4)
            padc=ROUNDUP4(c);
        else
            padc=c;

        size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
    }

    Blob::Blob(int height,int width,int channel,std::string name,float* data_in)
    {
        name=name;

        h=height;
        w=width;
        c=channel;
        if(c>4)
            padc=ROUNDUP4(c);
        else
            padc=c;
        this->data_ptr=data_in;
        size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
    }

    int Blob::create(int height,int width,int channel,std::string name)
    {
        name=name;
        h=height;
        w=width;
        c=channel;
        if(c>4)
            padc=ROUNDUP4(c);
        else
            padc=c;
        data_ptr=fastnn_alloc(w*h*padc* sizeof(float));

        size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
        owner=true;
    }

    int Blob::setData(float* data_in)
    {
        if(data_in!=NULL)
        {
            this->data_ptr=data_in;
            size=w*h*padc;
        }
        else
            return -1;
        return 0;
    }

    int Blob::setSize( std::vector<int> size)
    {
        if(size.size() ==4)
        {
            c = size[3];
            w = size[2];
            h = size[1];
            if (c > 4)
                padc = ROUNDUP4(c);
            else
                padc = c;

            this->size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
            return 0;
        }
        return -1;
    }

    int Blob::setSize(int *size)
    {
        if(size!=NULL)
        {
            c = size[3];
            w = size[2];
            h = size[1];
            if (c > 4)
                padc = ROUNDUP4(c);
            else
                padc = c;
        }
        this->size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
        return 0;
    }

    int Blob::setSize(int height,int width,int channel)
    {
        h=height;
        w=width;
        c=channel;
        if(c>4)
            padc=ROUNDUP4(c);
        else
            padc=c;

        this->size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
        return 0;
    }

    int Blob::get_blob_shape(int *ptr)
    {
        if(ptr!=NULL)
        {
            ptr[0]=1;
            ptr[1]=h;
            ptr[2]=w;
            ptr[3]=c;

            if(c>4)
                padc=ROUNDUP4(c);
            else
                padc=c;
            return 0;
        }
        else
        {
            return -1;
        }
    }

    int Blob::dim()
    {
        int dim=3;
        if(c<=1)
        {
            dim-=1;
            if(h<=1)
            {
                dim-=1;
                if(w<=1)
                {
                    dim=0;
                }
            }

        }
        return dim;
    }

    float* Blob::data()
    {
        if(size>0 && data_ptr!=NULL)
        {
            return this->data_ptr;
        }else
        {
            return 0;
        }
    }

    int Blob::set_padparam(int up,int down,int left,int right)
    {
        pad_up=up;
        pad_down=down;
        pad_left=left;
        pad_right=right;

        this->size=(w+pad_left+pad_right)*(h+pad_up+pad_down)*padc;
        return 0;
    }

    int Blob::set_pad_zero()
    {
        int channe_out=c;
        float * dst=data();

        if (pad_right || pad_left || pad_up || pad_down)
        {
            int pad_width = w + pad_right + pad_left;
            int pad_height = h + pad_up + pad_down;
            int pad_channel = padc;

            memset(dst, 0.0f, (pad_up * pad_width + pad_left) * pad_channel * sizeof(float));
            dst += ((pad_up + 1) * pad_width - pad_right) * pad_channel;
            for (int height = 0; height < h - 1; height++)
            {
                memset(dst, 0.0f, (pad_width - w) * pad_channel * sizeof(float));
                dst += pad_width * pad_channel;
            }
            memset(dst, 0.0f, (pad_down * pad_width + pad_right) * pad_channel * sizeof(float));
        }
        return 0;
    }

} // namespace fastnn
