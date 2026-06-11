NVCC = nvcc
CUDA_HOME ?= /usr/local/cuda
GM_DIR = ../gm_graph
GM_INC = $(GM_DIR)/inc
GM_LIB = $(GM_DIR)/lib

NVCCFLAGS = -O3 -I. -I$(GM_INC) -std=c++11 -arch=sm_70
LFLAGS   = -L$(GM_LIB) -lgmgraph -L$(CUDA_HOME)/lib64 -lcudart

TARGET = scc_cuda
OBJS = scc_cuda_main.o scc_cuda_graph.o scc_cuda_trim1.o scc_cuda_trim2.o scc_cuda_trim2_new.o

all: $(TARGET)

$(TARGET): $(OBJS) $(GM_LIB)/libgmgraph.a
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LFLAGS) -o $@

scc_cuda_main.o: scc_cuda_main.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_graph.o: scc_cuda_graph.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_trim1.o: scc_cuda_trim1.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_trim2.o: scc_cuda_trim2.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

scc_cuda_trim2_new.o: scc_cuda_trim2_new.cu scc_cuda.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

clean:
	rm -f $(TARGET) *.o
