################################################################################
# Makefile for MNIST CUDNN Implementation (All Versions)
################################################################################

# Target executables
TARGET_BASE := bin/mnistCUDNN
TARGET_SIMPLE := bin/mnistCUDNN_simple
TARGET_SIMPLE_OPT := bin/mnistCUDNN_simple_optimized
TARGET_SIMPLE_NO_PREPROCESS := bin/mnistCUDNN_simple_no_preprocess
TARGET_IMPROVED := bin/mnistCUDNN_improved

# Source files
SOURCES_BASE := src/mnistCUDNN.cpp
SOURCES_SIMPLE := src/mnistCUDNN_simple.cpp
SOURCES_SIMPLE_OPT := src/mnistCUDNN_simple_optimized.cpp
SOURCES_SIMPLE_NO_PREPROCESS := src/mnistCUDNN_simple_no_preprocess.cpp
SOURCES_IMPROVED := src/mnistCUDNN_improved.cpp

# Object files
OBJECTS_BASE := obj/mnistCUDNN.o
OBJECTS_SIMPLE := obj/mnistCUDNN_simple.o
OBJECTS_SIMPLE_OPT := obj/mnistCUDNN_simple_optimized.o
OBJECTS_SIMPLE_NO_PREPROCESS := obj/mnistCUDNN_simple_no_preprocess.o
OBJECTS_IMPROVED := obj/mnistCUDNN_improved.o

# CUDA files
CUDA_SOURCES := src/fp16_dev.cu
CUDA_OBJECTS := obj/fp16_dev.o

# Common object files
COMMON_OBJECTS := obj/fp16_emu.o

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
all: $(TARGET_BASE) $(TARGET_SIMPLE) $(TARGET_SIMPLE_OPT) $(TARGET_SIMPLE_NO_PREPROCESS) $(TARGET_IMPROVED)

# Create directories if they don't exist
$(shell mkdir -p obj bin)

# Rule for base .cpp files
$(OBJECTS_BASE): $(SOURCES_BASE)
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for simple .cpp files
$(OBJECTS_SIMPLE): $(SOURCES_SIMPLE)
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for simple optimized .cpp files
$(OBJECTS_SIMPLE_OPT): $(SOURCES_SIMPLE_OPT)
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for simple no preprocess .cpp files
$(OBJECTS_SIMPLE_NO_PREPROCESS): $(SOURCES_SIMPLE_NO_PREPROCESS)
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for improved .cpp files
$(OBJECTS_IMPROVED): $(SOURCES_IMPROVED)
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Rule for .cu files
$(CUDA_OBJECTS): $(CUDA_SOURCES)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GENCODE_FLAGS) -c $< -o $@

# Rule for fp16_emu.cpp
$(COMMON_OBJECTS): src/fp16_emu.cpp
	$(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link targets
$(TARGET_BASE): $(OBJECTS_BASE) $(CUDA_OBJECTS) $(COMMON_OBJECTS)
	$(HOST_COMPILER) $^ -o $@ $(LDFLAGS)

$(TARGET_SIMPLE): $(OBJECTS_SIMPLE) $(CUDA_OBJECTS) $(COMMON_OBJECTS)
	$(HOST_COMPILER) $^ -o $@ $(LDFLAGS)

$(TARGET_SIMPLE_OPT): $(OBJECTS_SIMPLE_OPT) $(CUDA_OBJECTS) $(COMMON_OBJECTS)
	$(HOST_COMPILER) $^ -o $@ $(LDFLAGS)

$(TARGET_SIMPLE_NO_PREPROCESS): $(OBJECTS_SIMPLE_NO_PREPROCESS) $(CUDA_OBJECTS) $(COMMON_OBJECTS)
	$(HOST_COMPILER) $^ -o $@ $(LDFLAGS)

$(TARGET_IMPROVED): $(OBJECTS_IMPROVED) $(CUDA_OBJECTS) $(COMMON_OBJECTS)
	$(HOST_COMPILER) $^ -o $@ $(LDFLAGS)

# Individual targets
base: $(TARGET_BASE)
simple: $(TARGET_SIMPLE)
simple-opt: $(TARGET_SIMPLE_OPT)
simple-no-preprocess: $(TARGET_SIMPLE_NO_PREPROCESS)
improved: $(TARGET_IMPROVED)

# Clean
clean:
	rm -f $(TARGET_BASE) $(TARGET_SIMPLE) $(TARGET_SIMPLE_OPT) $(TARGET_SIMPLE_NO_PREPROCESS) $(TARGET_IMPROVED) $(OBJECTS_BASE) $(OBJECTS_SIMPLE) $(OBJECTS_SIMPLE_OPT) $(OBJECTS_SIMPLE_NO_PREPROCESS) $(OBJECTS_IMPROVED) $(CUDA_OBJECTS) $(COMMON_OBJECTS)

# Install (copy to /usr/local/bin)
install: all
	cp $(TARGET_BASE) $(TARGET_SIMPLE) $(TARGET_SIMPLE_OPT) $(TARGET_SIMPLE_NO_PREPROCESS) $(TARGET_IMPROVED) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/mnistCUDNN /usr/local/bin/mnistCUDNN_simple /usr/local/bin/mnistCUDNN_simple_optimized /usr/local/bin/mnistCUDNN_simple_no_preprocess /usr/local/bin/mnistCUDNN_improved

# Help
help:
	@echo "Available targets:"
	@echo "  all                  - Build all MNIST CUDNN executables"
	@echo "  base                 - Build base MNIST CUDNN executable"
	@echo "  simple               - Build simple MNIST CUDNN executable"
	@echo "  simple-opt           - Build optimized simple MNIST CUDNN executable"
	@echo "  simple-no-preprocess - Build simple MNIST CUDNN executable without preprocessing"
	@echo "  improved             - Build improved MNIST CUDNN executable"
	@echo "  clean                - Remove all object files and executables"
	@echo "  install              - Install executables to /usr/local/bin"
	@echo "  uninstall            - Remove executables from /usr/local/bin"
	@echo "  help                 - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1   - Build with debug symbols"
	@echo "  CUDA_PATH - Path to CUDA installation (default: /usr/local/cuda)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make base simple improved"
	@echo "  make all"
	@echo "  make DEBUG=1 simple"
	@echo "  make clean"

# Phony targets
.PHONY: all base simple simple-opt simple-no-preprocess improved clean install uninstall help 