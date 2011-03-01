CC=clang++

all: generate_hmm.bin grabcut.bin

generate_hmm.bin: hmm.o main.o gaussian.o
	clang++ -g `pkg-config opencv --libs` gaussian.o hmm.o main.o -o generate_hmm.bin

gaussian.o: gaussian.cpp
	clang++ -g `pkg-config opencv --cflags` -c gaussian.cpp

hmm.o: hmm.cpp
	clang++ -g `pkg-config opencv --cflags` -c hmm.cpp

main.o: main.cpp
	clang++ -g `pkg-config opencv --cflags` -c main.cpp

grabcut.bin: grabcut.o sampleapp.o hmm.o gaussian.o
	clang++ -g `pkg-config opencv --cflags --libs` gaussian.o hmm.o grabcut.cpp sampleapp.cpp -o grabcut.bin

grabcut.o: grabcut.cpp
	clang++ -g `pkg-config opencv --cflags` -c grabcut.cpp

sampleapp.o: sampleapp.cpp
	clang++ -g `pkg-config opencv --cflags` -c sampleapp.cpp

clean:
	rm -rf *o *.bin 
