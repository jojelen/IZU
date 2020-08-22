# Choose your C++ compiler here (in general g++ on Linux systems):
CXX = g++
# Optimisation level, eg: -O3
OPT=-O3
# Compiler flags
CXXFLAGS=-std=c++17 -Wall $(OPT) -w -pthread -DTIME

# Location of .obj and .a files
BINDIR=build
OBJDIR=build/obj
LIBDIR=build/lib

# Includes
CXXFLAGS+=-Isrc

# New folders to be created
MKDIR_P = mkdir -p
NEWFOLDERS=build $(LIBDIR) $(OBJDIR)

SRC=$(subst src/, , $(wildcard src/*.cpp))
OBJ=$(SRC:.cpp=.o)
VPATH=src src/prog

#LDFLAGS is used for programs using created library in lib.
LDFLAGS=-L$(LIBDIR)

# OpenCV
CXXFLAGS+=-I/usr/local/include/opencv4
LDFLAGS+=-L/usr/local/lib `pkg-config --cflags --libs opencv`

# Tensorflow Lite
CXXFLAGS+=-Ideps/include
LDFLAGS+=-Ldeps/lib -ltensorflowlite

# Tensorflow Lite GPU delegate
LDFLAGS+=-ltensorflowlite_gpu_gl `pkg-config --cflags --libs egl glesv2`

LIBNAME=IZU
LIBS=lib$(LIBNAME).a
PROG=tflitex main

.PHONY: lib clean cleanall

all: createFolders lib $(PROG)

createFolders: $(NEWFOLDERS)

# Creating folders in OUT_DIR
$(NEWFOLDERS):
	@ $(MKDIR_P) $(NEWFOLDERS)

#Compiler command for any .cpp .h files
$(OBJDIR)/%.o : %.cpp %.h
	$(CXX) -c $< $(CXXFLAGS) -o $@

lib: $(addprefix $(LIBDIR)/, lib$(LIBNAME).a)

$(addprefix $(LIBDIR)/, lib$(LIBNAME).a): $(addprefix $(OBJDIR)/, $(OBJ))
	@ echo "Making $(LIBNAME) library in $@"
	@ ar rcs $@ $(addprefix $(OBJDIR)/, $(OBJ))

# Compiler command for executables
%: %.cpp $(addprefix $(LIBDIR)/, $(LIBS))
	$(CXX) $< $(CXXFLAGS) $(addprefix $(LIBDIR)/, $(LIBS)) $(LDFLAGS) -o $(addprefix $(BINDIR)/, $@)

clean:
	@ echo "Cleaning library"
	@ rm -f $(addprefix $(OBJDIR)/, *.o)
	@ rm -f $(addprefix $(LIBDIR)/, $(LIB))

cleanall:
	@ make -s clean
	@ echo "Cleaning build directory"
	@ rm -rf build
