cmake_minimum_required (VERSION 2.6)
project (HMM)

add_subdirectory(lib)
add_subdirectory(hmm)
add_subdirectory(naive)
add_subdirectory(st-naive)
add_subdirectory(msst-naive)
add_subdirectory(msst-hmm)
add_subdirectory(st-test)
add_subdirectory(st-interactive)
add_subdirectory(msst-interactive)
add_subdirectory(tests)
add_subdirectory(refine-mscr)
add_subdirectory(nldif)
add_subdirectory(outlines2masks)


