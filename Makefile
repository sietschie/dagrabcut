CC=clang++
CFLAGS= -g `pkg-config opencv --cflags`
LDFLAGS= `pkg-config opencv --libs`

all: generate_hmm.bin grabcut.bin

generate_hmm.bin: hmm.o main.o gaussian.o
	$(CC) $(LDFLAGS) $^ -o generate_hmm.bin

grabcut.bin: grabcut.o sampleapp.o hmm.o gaussian.o
	$(CC) $(LDFLAGS) $^ -o grabcut.bin

.cpp.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf *o *.bin 
