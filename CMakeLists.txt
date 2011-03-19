cmake_minimum_required (VERSION 2.6)
project (HMM)

FIND_PACKAGE(Boost COMPONENTS program_options unit_test_framework REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

add_executable(learn.bin learnmain.cpp hmm.cpp gmm.cpp gaussian.cpp grabcut.cpp graph.cpp shared.cpp maxflow.cpp)
target_link_libraries(learn.bin ${OpenCV_LIBS})
target_link_libraries(learn.bin ${Boost_LIBRARIES})

add_executable(test.bin testmain.cpp hmm.cpp gmm.cpp gaussian.cpp grabcut.cpp graph.cpp shared.cpp maxflow.cpp)
target_link_libraries(test.bin ${OpenCV_LIBS})
target_link_libraries(test.bin ${Boost_LIBRARIES})

add_executable(unittest.bin tests/test.cpp hmm.cpp gmm.cpp gaussian.cpp grabcut.cpp graph.cpp shared.cpp maxflow.cpp)
target_link_libraries(unittest.bin ${OpenCV_LIBS})
target_link_libraries(unittest.bin ${Boost_LIBRARIES})

ENABLE_TESTING()
ADD_TEST(Unittest unittest.bin)

