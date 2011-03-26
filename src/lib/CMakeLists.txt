FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

SET(mylib_SRCS hmm.cpp gmm.cpp gaussian.cpp grabcut.cpp maxflow/graph.cpp shared.cpp maxflow/maxflow.cpp structuretensor.cpp)
add_library( mylib ${mylib_SRCS})
target_link_libraries(mylib ${OpenCV_LIBS})
