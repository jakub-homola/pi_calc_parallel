#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "timer.h"



#define CHECK(status) do { check((status), __FILE__, __LINE__); } while(false)
void check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d %s: %s. In file '%s' on line %d\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), file, line);
    }
}

__global__ void calc_pi_kernel(size_t count, double * d_total_area)
{
    __shared__ double area_block;

    double interval_size = 1.0 / count;

    double area_my_thread = 0;

    size_t thread_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for(size_t i = thread_idx; i < count; i += stride)
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
    size_t count = 1000000000;
    if(argc > 1) count = atoll(argv[1]);

    printf("Number of intervals: %zu = %zu million\n", count, count / 1000000);
    printf("\n");
    printf("Calculating PI on Nvidia GPU\n");
    printf("\n");
    printf("...\n");

    CHECK(cudaSetDevice(0));



    Timer timer;
    timer.start();

    double total_area = 0;
    double * d_total_area;
    CHECK(cudaMalloc(&d_total_area, sizeof(double)));

    CHECK(cudaMemcpy(d_total_area, &total_area, sizeof(double), cudaMemcpyDefault));

    calc_pi_kernel<<< 108*16, 1024 >>>(count, d_total_area);

    CHECK(cudaMemcpy(&total_area, d_total_area, sizeof(double), cudaMemcpyDefault));

    double pi_calculated = total_area;

    timer.stop();



    CHECK(cudaFree(d_total_area));

    printf("\n");
    printf("Actual value of PI:     3.1415926535897932...\n");
    printf("\n");
    printf("Calculated value of PI: %.16f\n", pi_calculated);
    printf("\n");
    printf("Execution time: %10.3f ms\n", timer.get_seconds() * 1000);

    return 0;
}
