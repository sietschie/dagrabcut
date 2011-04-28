FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( outlines2mask_SRCS main.cpp )
add_executable(outlines2mask.bin ${outlines2mask_SRCS})
target_link_libraries(outlines2mask.bin ${OpenCV_LIBS})

