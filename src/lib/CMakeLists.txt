FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

SET(mylib_SRCS hmm.cpp gmm.cpp gaussian.cpp grabcut.cpp maxflow/graph.cpp shared.cpp maxflow/maxflow.cpp structuretensor.cpp st_gmm.cpp st_gaussian.cpp st_grabcut.cpp msst_grabcut.cpp msst_gmm.cpp cmsst_grabcut.cpp msst_hmm.cpp)
add_library( mylib ${mylib_SRCS})
target_link_libraries(mylib ${OpenCV_LIBS})
