CC= g++
NAME=convert
INCDIR = ../gm_graph/inc
LIBDIR = ../gm_graph/lib
CFLAGS = -g -O3 -fopenmp -I$(INCDIR) -I. 
LFLAGS = -L$(LIBDIR) -lgmgraph

$(NAME) : convert.cc convert.h
	$(CC) $(CFLAGS) $< $(LFLAGS) -o $@

clean:
	\rm $(NAME)
