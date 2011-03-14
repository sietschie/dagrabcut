program_NAME := learn.bin
program_OBJS := grabcut.o grabcutmain.o gaussian.o

program2_NAME := test.bin
program2_OBJS := grabcut.o testmain.o gaussian.o

CC=g++

CFLAGS+= -g `pkg-config opencv --cflags`
LDFLAGS+= `pkg-config opencv --libs`

all: $(program_NAME) $(program2_NAME)

$(program_NAME): $(program_OBJS)
	$(CC) $(LDFLAGS) $^ -o $(program_NAME)

$(program2_NAME): $(program2_OBJS)
	$(CC) $(LDFLAGS) $^ -o $(program2_NAME)

.cpp.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf *o *.bin 
