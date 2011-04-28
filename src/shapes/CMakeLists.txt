FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( shape_SRCS main.cpp )
add_executable(shape.bin ${shape_SRCS})
target_link_libraries(shape.bin ${OpenCV_LIBS})
target_link_libraries(shape.bin ${Boost_LIBRARIES})
target_link_libraries(shape.bin mylib)


