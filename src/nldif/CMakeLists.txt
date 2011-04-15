FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

set( nldif_SRCS main.cpp nldiff.cpp)
add_executable(nldif.bin ${nldif_SRCS})
target_link_libraries(nldif.bin ${OpenCV_LIBS})

