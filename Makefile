program_NAME := generate_hmm.bin
program_OBJS := hmm.o generate_hmmmain.o gaussian.o

program2_NAME := grabcut.bin
program2_OBJS := grabcut.o grabcutmain.o hmm.o gaussian.o

CC=clang++
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
