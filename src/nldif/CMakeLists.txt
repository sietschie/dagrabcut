FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( nldiff_SRCS main.cpp )
add_executable(nldiff.bin ${nldiff_SRCS})
target_link_libraries(nldiff.bin ${OpenCV_LIBS})
target_link_libraries(nldiff.bin mylib)


