# on Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS. The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. On Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU: type "module swap PrgEnv-pgi PrgEnv-gnu"

ifdef UBUNTU
  CC = gcc -D_GNU_SOURCE=1
  LDLIBS = -lrt -lblas -lm
  OSFOUND=1
endif
ifdef OSX
  CC = gcc
  LDLIBS = -lblas
  OSFOUND=1
endif
ifndef OSFOUND
  HOPPER = 1
  CC = cc
  LDLIBS = -lrt
endif

ifdef DEBUG
  INSTRUMENTATION = -g
endif
ifdef PROFILE
  INSTRUMENTATION = -pg -g
endif

ifdef NO_OPT
  OPTIMIZATION = -O0
else  
  OPTIMIZATION = -O3
endif

OPT = $(INSTRUMENTATION) $(OPTIMIZATION)

ifdef EXTERNAL_CFLAGS
  CFLAGS = -Wall -std=gnu99 $(OPT) $(EXTERNAL_CFLAGS)
else
  CFLAGS = -Wall -std=gnu99 $(OPT)
endif
LDFLAGS = -Wall

OWN_LIBS = unit-test-framework.o matrix-blocking.o matrix-storage.o

targets = benchmark-naive benchmark-blocked-baseline benchmark-blas benchmark-simd benchmark-blocked-simple test-blocked-multi test-blocked-simple
objects = benchmark.o $(OWN_LIBS) dgemm-blocked-multi-tests.o dgemm-blocked-simple-tests.o dgemm-naive.o dgemm-blocked-baseline.o dgemm-blas.o dgemm-blocked-multi.o dgemm-simd.o dgemm-blocked-simple.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-baseline : benchmark.o dgemm-blocked-baseline.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked-multi : benchmark.o dgemm-blocked-multi.o $(OWN_LIBS)
	$(CC) -o $@ $^ $(LDLIBS)
test-blocked-multi : dgemm-blocked-multi-tests.o dgemm-blocked-multi.o $(OWN_LIBS)
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-simd : benchmark.o dgemm-simd.o
	$(CC) -DCLS=$(getconf LEVEL1_DCACHE_LINESIZE) -O3 -o $@ $^ $(LDLIBS)
benchmark-blocked-simple : benchmark.o dgemm-blocked-simple.o $(OWN_LIBS)
	$(CC) -o $@ $^ $(LDLIBS)
test-blocked-simple : dgemm-blocked-simple-tests.o dgemm-blocked-simple.o $(OWN_LIBS)
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
