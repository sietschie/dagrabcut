FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( MSST_interactive_SRCS sampleapp.cpp )
add_executable(MSST_interactive.bin ${MSST_interactive_SRCS})
target_link_libraries(MSST_interactive.bin ${OpenCV_LIBS})
target_link_libraries(MSST_interactive.bin mylib)

