NVCC = nvcc
CXX  = g++
CUDA_HOME ?= /usr/local/cuda
GM_DIR = ../gm_graph
GM_INC = $(GM_DIR)/inc
GM_LIB = $(GM_DIR)/lib

# Auto-detect GPU architecture from nvidia-smi
GPU_NAME := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ifeq ($(GPU_NAME),)
  ARCH = sm_70
else ifneq (,$(findstring A100,$(GPU_NAME)))
  ARCH = sm_80
else ifneq (,$(findstring V100,$(GPU_NAME)))
  ARCH = sm_70
else ifneq (,$(findstring T4,$(GPU_NAME)))
  ARCH = sm_75
else ifneq (,$(findstring RTX 40,$(GPU_NAME)))
  ARCH = sm_89
else ifneq (,$(findstring RTX 30,$(GPU_NAME)))
  ARCH = sm_86
else ifneq (,$(findstring RTX 20,$(GPU_NAME)))
  ARCH = sm_75
else ifneq (,$(findstring P100,$(GPU_NAME)))
  ARCH = sm_60
else ifneq (,$(findstring K80,$(GPU_NAME)))
  ARCH = sm_37
else
  # Default to sm_89 for newer GPUs (Ada/RTX 40 series)
  ARCH = sm_89
endif

NVCCFLAGS = -O3 -I. -I$(GM_INC) -std=c++11 -arch=$(ARCH) -Xcompiler -fopenmp
CXXFLAGS  = -O3 -I. -I$(GM_INC) -I$(CUDA_HOME)/include -std=c++11 -fopenmp
LFLAGS    = -L$(CUDA_HOME)/lib64 -lcudart -L$(GM_LIB) -lgmgraph -lgomp

TARGET = scc_cuda
OBJS = scc_cuda_main.o scc_cuda_dynamic.o scc_cuda_graph.o scc_cuda_trim1.o scc_cuda_trim2.o scc_cuda_trim2_new.o scc_cuda_fb_global.o scc_cuda_fb_seq.o scc_cuda_fb_seq2.o scc_cuda_weak.o scc_cuda_work_queue.o

all: $(TARGET)

$(TARGET): $(OBJS) $(GM_LIB)/libgmgraph.a
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LFLAGS) -o $@

# Host-only: compiled with g++ (gm_graph is not CUDA-compatible)
scc_cuda_main.o: scc_cuda_main.cpp scc_cuda.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Dynamic graph construction helpers (host-only, uses gm_graph)
scc_cuda_dynamic.o: scc_cuda_dynamic.cpp scc_cuda.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA device code: compiled with nvcc -dc
scc_cuda_graph.o: scc_cuda_graph.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_trim1.o: scc_cuda_trim1.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_trim2.o: scc_cuda_trim2.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_trim2_new.o: scc_cuda_trim2_new.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_fb_global.o: scc_cuda_fb_global.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_fb_seq.o: scc_cuda_fb_seq.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_fb_seq2.o: scc_cuda_fb_seq2.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_weak.o: scc_cuda_weak.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_work_queue.o: scc_cuda_work_queue.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -f $(TARGET) *.o
