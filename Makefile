all:
	g++ `pkg-config opencv --cflags --libs` hmm.cpp main.cpp -o prog.bin
