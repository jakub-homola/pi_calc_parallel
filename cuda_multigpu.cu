#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include "timer.h"



#define CHECK(status) do { check((status), __FILE__, __LINE__); } while(false)
void check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d %s: %s. In file '%s' on line %d\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), file, line);
    }
}

__global__ void calc_pi_kernel(size_t i_start, size_t i_end, double interval_size, double * d_total_area)
{
    __shared__ double area_block;

    double area_my_thread = 0;

    size_t thread_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for(size_t i = i_start + thread_idx; i < i_end; i += stride)
    {
        double x = (i+0.5) * interval_size;

        double value = 4 / (1 + x * x);
        double area = value * interval_size;

        area_my_thread += area;
    }

    if(threadIdx.x == 0) area_block = 0;
    __syncthreads();
    atomicAdd_block(&area_block, area_my_thread);
    __syncthreads();
    if(threadIdx.x == 0) atomicAdd(d_total_area, area_block);
}



int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    size_t count = 1000000000;
    if(argc > 1) count = atoll(argv[1]);

    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if(mpi_rank == 0)
    {
        printf("Number of intervals: %zu = %zu million\n", count, count / 1000000);
        printf("\n");
        printf("Calculating PI on %d Nvidia GPUs\n", mpi_rank);
        printf("\n");
        printf("...\n");
    }

    int mpi_gpu_map[] = {2,3,0,1,6,7,4,5};
    CHECK(cudaSetDevice(mpi_gpu_map[mpi_rank % 8]));



    Timer timer;
    timer.start();

    double * d_total_area;
    CHECK(cudaMalloc(&d_total_area, sizeof(double)));

    double interval_size = 1.0 / count;
    double total_area_my_process = 0;

    size_t count_per_process = count / mpi_size;
    size_t leftover = count % mpi_size;
    size_t my_i_start = mpi_rank * count_per_process + std::min<size_t>(mpi_rank, leftover);
    size_t my_i_end = (mpi_rank + 1) * count_per_process + std::min<size_t>(mpi_rank + 1, leftover);

    CHECK(cudaMemcpy(d_total_area, &total_area_my_process, sizeof(double), cudaMemcpyDefault));

    calc_pi_kernel<<< 108*16, 1024 >>>(my_i_start, my_i_end, interval_size, d_total_area);

    CHECK(cudaMemcpy(&total_area_my_process, d_total_area, sizeof(double), cudaMemcpyDefault));

    double total_area;
    MPI_Reduce(&total_area_my_process, &total_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi_calculated = total_area;

    timer.stop();



    CHECK(cudaFree(d_total_area));

    if(mpi_rank == 0)
    {
        printf("\n");
        printf("Actual value of PI:     3.1415926535897932...\n");
        printf("\n");
        printf("Calculated value of PI: %.16f\n", pi_calculated);
        printf("\n");
        printf("Execution time: %10.3f ms\n", timer.get_seconds() * 1000);
    }

    MPI_Finalize();

    return 0;
}
