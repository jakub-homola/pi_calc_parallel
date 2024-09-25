
CXX=g++
CXX_FLAGS=-g -O2

.PHONY: clean



clean:
	rm -f *.x



cpp_sequential.x: cpp_sequential.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} $< -o $@

cpp_openmp.x: cpp_openmp.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} -fopenmp $< -o $@

cpp_openmp.x: cpp_openmp.cpp timer.h Makefile
	${CXX} ${CXX_FLAGS} $< -o $@ -lmpi
