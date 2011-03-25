FIND_PACKAGE(Boost COMPONENTS unit_test_framework REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( unittest_SRCS test.cpp test_hmm.cpp )
add_executable(unittest.bin ${unittest_SRCS} )
target_link_libraries(unittest.bin ${OpenCV_LIBS})
target_link_libraries(unittest.bin ${Boost_LIBRARIES})
target_link_libraries(unittest.bin mylib)


ENABLE_TESTING()
ADD_TEST(Unittest unittest.bin)

