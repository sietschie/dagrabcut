FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( learn_SRCS learnmain.cpp )
add_executable(learn.bin ${learn_SRCS})
target_link_libraries(learn.bin ${OpenCV_LIBS})
target_link_libraries(learn.bin ${Boost_LIBRARIES})
target_link_libraries(learn.bin mylib)

set( test_SRCS testmain.cpp )
add_executable(test.bin ${test_SRCS})
target_link_libraries(test.bin ${OpenCV_LIBS})
target_link_libraries(test.bin ${Boost_LIBRARIES})
target_link_libraries(test.bin mylib)

