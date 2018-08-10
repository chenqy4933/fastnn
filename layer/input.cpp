

#include "input.h"

namespace fastnn {


Input::Input()
{
    support_inplace = true;
}

int Input::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}
int Input::infershape()
{
    int top_size=tops.size();
    if(top_size>1)
    {
        printf("The input layer have two many top blobs\n");
    }
    else
    {
        std::vector<int> shape(4);
        shape[0]=1;
        shape[1]=h;
        shape[2]=w;
        shape[3]=c;
        tops[0]->setSize(shape);
    }
    return 0;
}
int Input::forward() const
{
    return 0;
}

} // namespace fastnn
