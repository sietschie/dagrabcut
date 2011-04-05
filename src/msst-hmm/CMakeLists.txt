FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( learn_SRCS learnmain.cpp )
add_executable(msst_hmm_learn.bin ${learn_SRCS})
target_link_libraries(msst_hmm_learn.bin ${OpenCV_LIBS})
target_link_libraries(msst_hmm_learn.bin ${Boost_LIBRARIES})
target_link_libraries(msst_hmm_learn.bin mylib)
set_target_properties(msst_hmm_learn.bin PROPERTIES OUTPUT_NAME learn.bin)

set( test_SRCS testmain.cpp )
add_executable(msst_hmm_test.bin ${test_SRCS})
target_link_libraries(msst_hmm_test.bin ${OpenCV_LIBS})
target_link_libraries(msst_hmm_test.bin ${Boost_LIBRARIES})
target_link_libraries(msst_hmm_test.bin mylib)
set_target_properties(msst_hmm_test.bin PROPERTIES OUTPUT_NAME test.bin)

