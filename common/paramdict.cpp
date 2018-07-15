

#include <ctype.h>
#include "paramdict.h"
#include "memoryAlloc.h"

namespace fastnn {

ParamDict::ParamDict()
{
    clear();
}

int ParamDict::get(int id, int def) const
{
    return params[id].loaded ? params[id].i : def;
}

float ParamDict::get(int id, float def) const
{
    return params[id].loaded ? params[id].f : def;
}

float * ParamDict::get(int id, float * def) const
{
    return params[id].loaded ? params[id].v : def;
}

void ParamDict::set(int id, int i)
{
    params[id].loaded = 1;
    params[id].i = i;
}

void ParamDict::set(int id, float f)
{
    params[id].loaded = 1;
    params[id].f = f;
}

void ParamDict::set(int id, float * v)
{
    params[id].loaded = 1;
    params[id].v = v;
}

void ParamDict::clear()
{
    for (int i = 0; i < MAX_PARAM_COUNT; i++)
    {
        params[i].loaded = 0;
        params[i].v = NULL;
    }
}

static bool vstr_is_float(const char vstr[16])
{
    // look ahead for determine isfloat
    for (int j=0; j<16; j++)
    {
        if (vstr[j] == '\0')
            break;

        if (vstr[j] == '.' || tolower(vstr[j]) == 'e')
            return true;
    }

    return false;
}

int ParamDict::load_param(FILE* fp)
{
    clear();

//     0=100 1=1.250000 -23303=5,0.1,0.2,0.4,0.8,1.0

    // parse each key=value pair
    int id = 0;
    while (fscanf(fp, "%d=", &id) == 1)
    {
        bool is_array = id <= -23300;
        if (is_array)
        {
            id = -id - 23300;
        }

        if (is_array)
        {
            int len = 0;
            int nscan = fscanf(fp, "%d", &len);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read array length fail\n");
                return -1;
            }

            params[id].v=fastnn_alloc(len);

            for (int j = 0; j < len; j++)
            {
                char vstr[16];
                nscan = fscanf(fp, ",%15[^,\n ]", vstr);
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict read array element fail\n");
                    return -1;
                }

                bool is_float = vstr_is_float(vstr);

                if (is_float)
                {
                    float* ptr = params[id].v;
                    nscan = sscanf(vstr, "%f", &ptr[j]);
                }
                else
                {
                    int* ptr = (int*)params[id].v;
                    nscan = sscanf(vstr, "%d", &ptr[j]);
                }
                if (nscan != 1)
                {
                    fprintf(stderr, "ParamDict parse array element fail\n");
                    return -1;
                }
            }
        }
        else
        {
            char vstr[16];
            int nscan = fscanf(fp, "%15s", vstr);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict read value fail\n");
                return -1;
            }

            bool is_float = vstr_is_float(vstr);

            if (is_float)
                nscan = sscanf(vstr, "%f", &params[id].f);
            else
                nscan = sscanf(vstr, "%d", &params[id].i);
            if (nscan != 1)
            {
                fprintf(stderr, "ParamDict parse value fail\n");
                return -1;
            }
        }

        params[id].loaded = 1;
    }

    return 0;
}


} // namespace fastnn
