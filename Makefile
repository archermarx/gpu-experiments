CC=nvcc
INC_DIR=include
LIBS=-lglfw -lGL -lX11 -lXi -lXrandr -ldl
CPPFLAGS=-I$(INC_DIR) -g
SRC_DIR=src
OBJ_DIR=build
EXE=run.exe

_DEPS=\
	glad/glad.h \
	cuda_helpers.h\
	utils.h \
	window.h \
	canvas.h \
	shader.h \
	color.h	\
	automaton.h \
	langton_ant.h \
	game_of_life.h

DEPS=$(addprefix $(INC_DIR)/, $(_DEPS))

_SRCS=\
	main.cu	\
	glad.c	\
	utils.cu	\
	window.cu  \
	canvas.cu 	\
	shader.cu	\
	langton_ant.cu \
	game_of_life.cu

SRCS=$(addprefix $(SRC_DIR)/, $(_SRCS))

_OBJS_C=$(patsubst %.c, %.o, $(_SRCS))
_OBJS=$(patsubst %.cu, %.o, $(_OBJS_C))
OBJS=$(addprefix $(OBJ_DIR)/, $(_OBJS))

# Recipes

all: prep $(EXE)

remake: clean all

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(DEPS)
	@$(CC) -c -o $@ $< $(CPPFLAGS)
	@echo $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(DEPS)
	@$(CC) -c -o $@ $< $(CPPFLAGS)
	@echo $<

$(EXE): $(OBJS)
	@$(CC) -o $@ $^ $(CPPFLAGS) $(LIBS)
	@echo $@

prep:
	@mkdir -p $(OBJ_DIR)

test: $(EXE)
	./$(EXE)

debug:
	@echo $(_OBJS)

.PHONY : clean

clean:
	@rm -f $(EXE) $(OBJ_DIR)/*.o