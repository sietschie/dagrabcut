CC=clang++

all: generate_hmm.bin grabcut.bin

generate_hmm.bin: hmm.o main.o
	clang++ `pkg-config opencv --libs` hmm.o main.o -o generate_hmm.bin

hmm.o: hmm.cpp
	clang++ `pkg-config opencv --cflags` -c hmm.cpp

main.o: main.cpp
	clang++ `pkg-config opencv --cflags` -c main.cpp

grabcut.bin: grabcut.o sampleapp.o
	clang++ `pkg-config opencv --cflags --libs` grabcut.cpp sampleapp.cpp -o grabcut.bin

grabcut.o: grabcut.cpp
	clang++ `pkg-config opencv --cflags` -c grabcut.cpp

sampleapp.o: sampleapp.cpp
	clang++ `pkg-config opencv --cflags` -c sampleapp.cpp

clean:
	rm -rf *o *.bin 
