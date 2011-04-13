FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( refine_SRCS grabcut.cpp )
add_executable(refine.bin ${refine_SRCS})
target_link_libraries(refine.bin ${OpenCV_LIBS})

