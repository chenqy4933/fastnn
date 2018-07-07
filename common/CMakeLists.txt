
##############################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

list(APPEND fastnn_SRCS
        layer.cpp
        blob.cpp
        mat.cpp
        modelbin.cpp
        net.cpp
        paramdict.cpp
)
