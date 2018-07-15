

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

int Input::forward(const std::vector<Blob> bottoms,std::vector<Blob> tops) const
{
    return 0;
}

} // namespace fastnn
