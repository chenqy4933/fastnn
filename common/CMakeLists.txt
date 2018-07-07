
##############################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(fastnn_SRCS
        layer.cpp
        blob.cpp
        mat.cpp
        modelbin.cpp
        net.cpp
        paramdict.cpp
)

add_library(fastnn STATIC ${fastnn_SRCS})

install(TARGETS fastnn ARCHIVE DESTINATION lib)
install(FILES
    blob.h
    mat.h
    met.h
    layer.h
    DESTINATION include
)
