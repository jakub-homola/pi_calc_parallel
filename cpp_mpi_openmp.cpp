#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include "timer.h"



int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    size_t count = 1000000000;
    if(argc > 1)
    {
        count = atoll(argv[1]);
        char last = *(strchr(argv[1], '\0') - 1);
        if(last == 'K') count *= 1000;
        if(last == 'M') count *= 1000000;
        if(last == 'B') count *= 1000000000;
    }

    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if(mpi_rank == 0)
    {
        char units = ' ';
        size_t count_to_print = count;
        if(count > 1000) { units = 'K'; count_to_print = count / 1000; }
        if(count > 1000000) { units = 'M'; count_to_print = count / 1000000; }
        if(count > 1000000000) { units = 'B'; count_to_print = count / 1000000000; }
        printf("Number of intervals: %zu = %zu %c\n", count, count_to_print, units);
        printf("\n");
        printf("Calculating PI using MPI with %d processes, each using OpenMP with %d threads\n", mpi_size, omp_get_max_threads());
        printf("\n");
        printf("...\n");
    }



    Timer timer;
    timer.start();

    double interval_size = 1.0 / count;
    double total_area_my_process = 0;

    size_t count_per_process = count / mpi_size;
    size_t leftover = count % mpi_size;
    size_t my_i_start = mpi_rank * count_per_process + std::min<size_t>(mpi_rank, leftover);
    size_t my_i_end = (mpi_rank + 1) * count_per_process + std::min<size_t>(mpi_rank + 1, leftover);

    #pragma omp parallel for reduction(+:total_area_my_process)
    for(size_t i = my_i_start; i < my_i_end; i++)
    {
        double x = (i+0.5) * interval_size;

        double value = 4 / (1 + x * x);
        double area = value * interval_size;

        total_area_my_process += area;
    }

    double total_area;
    MPI_Reduce(&total_area_my_process, &total_area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi_calculated = total_area;

    timer.stop();



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
