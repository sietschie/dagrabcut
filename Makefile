all:
	clang++ `pkg-config opencv --cflags --libs` hmm.cpp main.cpp -o prog.bin
