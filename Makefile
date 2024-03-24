CC=g++
INC_DIR=include
LIBS=-lglfw -lGL -lX11 -lXi -lpthread -lXrandr -ldl
CPPFLAGS=-pthread -I$(INC_DIR)
SRC_DIR=src
OBJ_DIR=build
EXE=run.exe

_DEPS=\
	glad/glad.h \
	utils.h \
	window.h \
	canvas.h \
	shader.h \
	langton_ant.h

DEPS=$(addprefix $(INC_DIR)/, $(_DEPS))

_SRCS=\
	main.cpp	\
	glad.cpp	\
	utils.cpp	\
	window.cpp  \
	canvas.cpp 	\
	shader.cpp	\
	langton_ant.cpp

SRCS=$(addprefix $(SRC_DIR)/, $(_SRCS))

_OBJS=$(patsubst %.cpp, %.o, $(_SRCS))
OBJS=$(addprefix $(OBJ_DIR)/, $(_OBJS))

# Recipes

all: prep $(EXE)

remake: clean all

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	@$(CC) -c -o $@ $< $(CPPFLAGS)
	@echo $<

$(EXE): $(OBJS)
	@$(CC) -o $@ $^ $(CPPFLAGS) $(LIBS)
	@echo $@

prep:
	@mkdir -p $(OBJ_DIR)

test:
	@echo $(OBJS)

.PHONY : clean

clean:
	@rm -f $(EXE) $(OBJ_DIR)/*.o