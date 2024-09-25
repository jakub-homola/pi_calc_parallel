
CXX=g++
CXX_FLAGS=-g -O2

.PHONY: clean



clean:
	rm -f *.x



cpp_sequential.x: cpp_sequential.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} $< -o $@

cpp_openmp.x: cpp_openmp.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} -fopenmp $< -o $@

cpp_mpi.x: cpp_mpi.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} $< -o $@ -lmpi_cxx -lmpi

cpp_mpi_openmp.x: cpp_mpi_openmp.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} -fopenmp $< -o $@ -lmpi_cxx -lmpi
