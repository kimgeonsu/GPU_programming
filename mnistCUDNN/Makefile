################################################################################
# Makefile for Improved MNIST CUDNN Implementation
################################################################################

# Target executable
TARGET := mnistCUDNN_improved

# Source files
SOURCES := mnistCUDNN_improved.cpp

# Object files
OBJECTS := $(SOURCES:.cpp=.o)

# CUDA files
CUDA_SOURCES := fp16_dev.cu
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)

# Host compiler
HOST_COMPILER ?= g++

# CUDA compiler
NVCC := nvcc

# Check CUDA installation
CUDA_PATH ?= /usr/local/cuda

# Architecture
HOST_ARCH := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)

# Operating system
ifeq ($(shell uname -s),Darwin)
    DARWIN := 1
endif

# Directories
INCLUDES := -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -IFreeImage/include
LIBRARIES := -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/samples/common/lib/linux/x86_64 -LFreeImage/lib

# Compiler flags
CXXFLAGS := -std=c++11 -O2
NVCCFLAGS := -std=c++11 -O2 -lineinfo

# Debug flags
ifdef DEBUG
    CXXFLAGS += -g -DDEBUG
    NVCCFLAGS += -g -G -DDEBUG
endif

# CUDA libraries
CUDA_LIBS := -lcudart -lcublas -lcudnn

# FreeImage library
FREEIMAGE_LIBS := -lfreeimage

# Check for 32-bit
ifeq ($(TARGET_ARCH),armv7l)
    LIBRARIES := -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/samples/common/lib/linux/armv7l -LFreeImage/lib
endif

# Gencode arguments (updated for newer CUDA versions)
ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),armv7l aarch64 sbsa))
    SMS ?= 50 52 60 61 70 72 75 80 86 87
else
    SMS ?= 50 52 60 61 70 75 80 86
endif

# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach SM,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(SM),code=sm_$(SM)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif

# Compiler-specific flags
LDFLAGS := $(LIBRARIES) $(CUDA_LIBS) $(FREEIMAGE_LIBS) -lm -ldl

# Default target
all: $(TARGET)

# Rule for .cpp files
%.o: %.cpp
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for .cu files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GENCODE_FLAGS) -c $< -o $@

# Link target
$(TARGET): $(OBJECTS) $(CUDA_OBJECTS) fp16_emu.o
	$(HOST_COMPILER) $(OBJECTS) $(CUDA_OBJECTS) fp16_emu.o -o $@ $(LDFLAGS)

# Compile fp16_emu.cpp if it doesn't exist as object
fp16_emu.o: fp16_emu.cpp
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c fp16_emu.cpp -o fp16_emu.o

# Clean
clean:
	rm -f $(TARGET) $(OBJECTS) $(CUDA_OBJECTS) fp16_emu.o

# Install (copy to /usr/local/bin)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/$(TARGET)

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build the improved MNIST CUDNN executable"
	@echo "  clean     - Remove all object files and executable"
	@echo "  install   - Install executable to /usr/local/bin"
	@echo "  uninstall - Remove executable from /usr/local/bin"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1   - Build with debug symbols"
	@echo "  CUDA_PATH - Path to CUDA installation (default: /usr/local/cuda)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make -f Makefile_improved"
	@echo "  make -f Makefile_improved DEBUG=1"
	@echo "  make -f Makefile_improved clean"

# Phony targets
.PHONY: all clean install uninstall help 