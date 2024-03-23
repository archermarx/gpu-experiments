CC=g++
INC_DIR=include
CFLAGS=-lglfw -lGL -lX11 -lXi -lpthread -lXrandr -ldl -I$(INC_DIR)
SRC_DIR=src
EXE=run.exe
SRCS=$(SRC_DIR)/learnOpenGL.cpp $(SRC_DIR)/glad.c

main: $(SRCS)
	$(CC) -pthread -o $(EXE) $(SRCS) $(CFLAGS)

clean:
	@rm -f $(EXE)