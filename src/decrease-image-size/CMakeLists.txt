FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( dis_SRCS main.cpp )
add_executable(dis.bin ${dis_SRCS})
target_link_libraries(dis.bin ${OpenCV_LIBS})

