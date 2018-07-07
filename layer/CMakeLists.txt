

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast -std=c++11 -Wno-format")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wall")





file(GLOB LAYER_SRC *.cpp)

list(APPEND fastnn_SRCS ${LAYER_SRC})

