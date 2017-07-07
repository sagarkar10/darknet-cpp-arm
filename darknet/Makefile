DEBUG=0
VPATH=./src/
EXEC=darknet
OBJDIR=./obj/
CC=gcc
# C++ Definitions
EXEC_CPP=darknet-cpp
OBJDIR_CPP=./obj-cpp/
CC_CPP=g++
CFLAGS_CPP=-Wno-write-strings -std=c++0x -I../ComputeLibrary/

OPTS=-O3 -funroll-loops -ftree-vectorize 

LDFLAGS=-lm ../ComputeLibrary/build/arm_compute/libarm_compute-static.a ../ComputeLibrary/build/arm_compute/libarm_compute_core-static.a -lpthread 
COMMON= 
CFLAGS=-Wall -Wfatal-errors 


ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ= utils.o image.o box.o TinyYolo.o 
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.hpp) Makefile

OBJS_CPP = $(addprefix $(OBJDIR_CPP), $(OBJ))

all: backup obj-cpp results $(EXEC_CPP)

$(EXEC_CPP): obj-cpp $(OBJS_CPP)
	$(CC_CPP) $(COMMON) $(CFLAGS_CPP) $(OBJS_CPP) -o $@ $(LDFLAGS)

$(OBJDIR_CPP)%.o: %.c $(DEPS)
	$(CC_CPP) $(COMMON) $(CFLAGS_CPP) $(CFLAGS) -c $< -o $@
$(OBJDIR_CPP)%.o: %.cpp $(DEPS)
	$(CC_CPP) $(COMMON) $(CFLAGS_CPP) $(CFLAGS) -c $< -o $@

obj:	
	mkdir -p obj
obj-cpp:
	mkdir -p obj-cpp

backup:
	mkdir -p backup

results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS_CPP) $(EXEC_CPP) 


