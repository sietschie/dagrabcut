program_NAME := learn.bin
program_OBJS := learnmain.o gaussian.o grabcut.o gmm.o shared.o hmm.o graph.o maxflow.o

program2_NAME := test.bin
program2_OBJS := grabcut.o testmain.o gaussian.o gmm.o shared.o hmm.o graph.o maxflow.o

CC=g++
CFLAGS+= -g `pkg-config opencv --cflags`
LDFLAGS+= `pkg-config opencv --libs` -lboost_program_options

all: $(program_NAME) $(program2_NAME)

$(program_NAME): $(program_OBJS)
	$(CC) $(LDFLAGS) $^ -o $(program_NAME)

$(program2_NAME): $(program2_OBJS)
	$(CC) $(LDFLAGS) $^ -o $(program2_NAME)

.cpp.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf *o *.bin 
