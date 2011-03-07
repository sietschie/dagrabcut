program_NAME := prog.bin
program_OBJS := grabcut.o grabcutmain.o gaussian.o

CC=clang++
CFLAGS+= -g `pkg-config opencv --cflags`
LDFLAGS+= `pkg-config opencv --libs`

all: $(program_NAME) $(program2_NAME)

$(program_NAME): $(program_OBJS)
	$(CC) $(LDFLAGS) $^ -o $(program_NAME)

.cpp.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf *o *.bin 
