
##############################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(fastnn_SRCS
    cpu.cpp
    layer.cpp
)


add_library(fastnn STATIC ${fastnn_SRCS})

install(TARGETS fastnn ARCHIVE DESTINATION lib)
install(FILES
    cpu.h
    layer.h

    DESTINATION include
)
