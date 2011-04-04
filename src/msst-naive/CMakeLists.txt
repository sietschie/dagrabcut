FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( msst_learn_SRCS learnmain.cpp )
add_executable(msst-naive-learn.bin ${msst_learn_SRCS})
target_link_libraries(msst-naive-learn.bin ${OpenCV_LIBS})
target_link_libraries(msst-naive-learn.bin ${Boost_LIBRARIES})
target_link_libraries(msst-naive-learn.bin mylib)
set_target_properties(msst-naive-learn.bin PROPERTIES OUTPUT_NAME learn.bin)

set( msst_test_SRCS testmain.cpp )
add_executable(msst-naive-test.bin ${msst_test_SRCS})
target_link_libraries(msst-naive-test.bin ${OpenCV_LIBS})
target_link_libraries(msst-naive-test.bin ${Boost_LIBRARIES})
target_link_libraries(msst-naive-test.bin mylib)
set_target_properties(msst-naive-test.bin PROPERTIES OUTPUT_NAME test.bin)

