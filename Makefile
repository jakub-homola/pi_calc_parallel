
CXX=g++
CXX_FLAGS=-g -O2

.PHONY: all clean



all: cpp_sequential.x cpp_openmp.x cpp_mpi.x cpp_mpi_openmp.x cuda_singlegpu.x cuda_multigpu.x

clean:
	rm -f *.x



cpp_sequential.x: cpp_sequential.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} $< -o $@

cpp_openmp.x: cpp_openmp.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} -fopenmp $< -o $@

cpp_mpi.x: cpp_mpi.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} $< -o $@ -lmpi -lmpi

cpp_mpi_openmp.x: cpp_mpi_openmp.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} -fopenmp $< -o $@ -lmpi -lmpi

cuda_singlegpu.x: cuda_singlegpu.cu timer.h Makefile
	nvcc ${CXX_FLAGS} -arch=native $< -o $@

cuda_multigpu.x: cuda_multigpu.cu timer.h Makefile
	nvcc ${CXX_FLAGS} -arch=native $< -o $@ -lmpi -lmpi
