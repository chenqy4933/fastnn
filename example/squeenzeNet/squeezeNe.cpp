//
// Created by 陈其友 on 2018/7/24.
//

#include "net.h"
#include "blob.h"
#include "stdio.h"

int squeezeNet_example(const char* param_path,const char* model_path)
{
    fastnn::NetEngine* net=new fastnn::NetEngine(param_path,model_path);

    net->init();

    int shape[4];
    net->get_input_shape(shape,"data");

    fastnn::Blob input_blob,out_blob;
    input_blob.create(shape[1],shape[2],shape[3],"data");
    //fill the input_blob with data


    //put the input blob to the Net
    net->input("data",input_blob);

    net->forward();

    net->output("prob",out_blob);
    return 0;
}
