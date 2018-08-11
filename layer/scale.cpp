

#include "scale.h"

namespace fastnn {


Scale::Scale()
{
    support_inplace = true;
}

Scale::~Scale()
{
    if (scale_data!=NULL) {
        fastnn_free(scale_data);
    }
    if (bias_data!=NULL) {
        fastnn_free(bias_data);
    }
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size != -233)
    {
        scale_data = mb.load(scale_data_size, 1);
        if (scale_data==NULL)
            return -100;
    }

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data==NULL)
            return -100;
    }

    return 0;
}


int Scale::forward() const
{

    int pad_left=tops[0]->pad_left;
    int pad_right=tops[0]->pad_right;
    int pad_down=tops[0]->pad_down;
    int pad_up=tops[0]->pad_up;

    float* pIn=bottoms[0]->data();
    float* pOut=tops[0]->data();

    int inw=bottoms[0]->w;
    int inh=bottoms[0]->h;
    int inc=bottoms[0]->c;
    int in_padc=bottoms[0]->padc;

    int dim=bottoms[0]->dim();

    if(dim==3)
    {
        if(bias_term)
        {
            if(pad_down || pad_left || pad_right ||pad_up)
            {
                float* in_end=pIn+inw*inh*in_padc;
                float* out_end=pOut+(inw+pad_left+pad_right)*(inh+pad_up+pad_down)*in_padc;
                for(int h=0;h<inh;h++)
                {
                    float* in=in_end-h*inw*in_padc;
                    float* out=out_end-h*(inw+pad_left+pad_right)*in_padc+pad_left*in_padc;
                    for(int w=0;w<inw;w++)
                    {
                        for(int c=0;c<inc;c++)
                            out[w]=scale_data[c]*in[w*in_padc+c]+bias_data[c];
                    }
                }
                return 0;
            }
            else
            {
                int size=inw*inh;
                for(int i=0;i<size;i++)
                {
                    float* in=pIn+i*in_padc;
                    float* out=pOut+i*in_padc;
                    for(int c=0;c<inc;c++)
                        out[i]=scale_data[c]*in[i]+bias_data[c];
                }
                return 0;
            }
        }
        else
        {
            if(pad_down || pad_left || pad_right ||pad_up)
            {
                float* in_end=pIn+inw*inh*in_padc;
                float* out_end=pOut+(inw+pad_left+pad_right)*(inh+pad_up+pad_down)*in_padc;
                for(int h=0;h<inh;h++)
                {
                    float* in=in_end-h*inw*in_padc;
                    float* out=out_end-h*(inw+pad_left+pad_right)*in_padc+pad_left*in_padc;
                    for(int w=0;w<inw;w++)
                    {
                        for(int c=0;c<inc;c++)
                            out[w]=scale_data[c]*in[w*in_padc+c];
                    }
                }
                return 0;
            }
            else
            {
                int size=inw*inh;
                for(int i=0;i<size;i++)
                {
                    float* in=pIn+i*in_padc;
                    float* out=pOut+i*in_padc;
                    for(int c=0;c<inc;c++)
                        out[i]=scale_data[c]*in[i];
                }
                return 0;
            }
        }
    }
    return -1;
}

} // namespace fastnn
