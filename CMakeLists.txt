cmake_minimum_required (VERSION 2.6)
project (HMM)

FIND_PACKAGE(Boost COMPONENTS program_options unit_test_framework REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

add_executable(learn.bin src/learnmain.cpp src/gmm.cpp src/gaussian.cpp src/grabcut.cpp src/maxflow/graph.cpp src/shared.cpp src/maxflow/maxflow.cpp)
target_link_libraries(learn.bin ${OpenCV_LIBS})
target_link_libraries(learn.bin ${Boost_LIBRARIES})

add_executable(test.bin src/testmain.cpp src/gmm.cpp src/gaussian.cpp src/grabcut.cpp src/maxflow/graph.cpp src/shared.cpp src/maxflow/maxflow.cpp)
target_link_libraries(test.bin ${OpenCV_LIBS})
target_link_libraries(test.bin ${Boost_LIBRARIES})

add_executable(unittest.bin tests/test.cpp src/gmm.cpp src/gaussian.cpp src/grabcut.cpp src/maxflow/graph.cpp src/shared.cpp src/maxflow/maxflow.cpp)
target_link_libraries(unittest.bin ${OpenCV_LIBS})
target_link_libraries(unittest.bin ${Boost_LIBRARIES})

ENABLE_TESTING()
ADD_TEST(Unittest unittest.bin)

