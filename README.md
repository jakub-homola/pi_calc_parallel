
# pi_calc_parallel



todo: image in readme and explanation

- sequential - OK
- pthreads
- OpenMP - OK
- MPI - OK
- MPI4Py - OK
- OpenMP + MPI - OK
- CUDA 1gpu
- CUDA multigpu 1 node loop
- CUDA multigpu 1 node OpenMP
- CUDA multigpu MPI
- HIP 1gpu
- HIP multigpu ...



each code will be a standalone program. No includes of common functions


cli args:
    total number of integration sections
print:
    total number of integration sections
    pi in double precision
    calculated value if pi
    total execution time





ml OpenMPI/4.1.6-NVHPC-24.1-CUDA-12.4.0

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=16,1

mpirun -n 4 --map-by ppr:1:socket --bind-to socket ./program.x

