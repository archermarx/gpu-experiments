CC=g++
INC_DIR=include
CFLAGS=-lglfw -lGL -lX11 -lXi -lpthread -lXrandr -ldl -I$(INC_DIR)
SRC_DIR=src
EXE=run.exe
SRCS=\
	main.cpp	\
	canvas.cpp 	\
	shader.cpp	\
	utils.cpp	\
	langton_ant.cpp \
	glad.c

SRC_FILES=$(addprefix $(SRC_DIR)/, $(SRCS))

main: $(SRC_FILES)
	$(CC) -pthread -o $(EXE) $(SRC_FILES) $(CFLAGS)

clean:
	@rm -f $(EXE)