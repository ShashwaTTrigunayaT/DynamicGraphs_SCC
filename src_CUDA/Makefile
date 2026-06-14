NVCC = nvcc
CXX  = g++
CUDA_HOME ?= /usr/local/cuda
GM_DIR = ../gm_graph
GM_INC = $(GM_DIR)/inc
GM_LIB = $(GM_DIR)/lib

NVCCFLAGS = -O3 -I. -I$(GM_INC) -std=c++11 -arch=sm_70
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
