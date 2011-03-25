all: test.bin

test.o: test.cpp
	gcc `pkg-config opencv --cflags` -c test.cpp 

test.bin: test.o
	gcc -lboost_unit_test_framework `pkg-config opencv --libs ` test.o ../gmm.o ../gaussian.o -o test.bin
