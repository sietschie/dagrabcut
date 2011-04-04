FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( sttest_SRCS main.cpp )
add_executable(sttest.bin ${sttest_SRCS})
target_link_libraries(sttest.bin ${OpenCV_LIBS})
target_link_libraries(sttest.bin mylib)


