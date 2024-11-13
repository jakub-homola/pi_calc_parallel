#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include "timer.h"



int main(int argc, char ** argv)
{
    size_t count = 1000000000;
    if(argc > 1)
    {
        count = atoll(argv[1]);
        char last = *(strchr(argv[1], '\0') - 1);
        if(last == 'K') count *= 1000;
        if(last == 'M') count *= 1000000;
        if(last == 'B') count *= 1000000000;
    }

    char units = ' ';
    size_t count_to_print = count;
    if(count > 1000) { units = 'K'; count_to_print = count / 1000; }
    if(count > 1000000) { units = 'M'; count_to_print = count / 1000000; }
    if(count > 1000000000) { units = 'B'; count_to_print = count / 1000000000; }
    printf("Number of intervals: %zu = %zu %c\n", count, count_to_print, units);
    printf("\n");
    printf("Calculating PI using OpenMP with %d threads\n", omp_get_max_threads());
    printf("\n");
    printf("...\n");



    Timer timer;
    timer.start();

    double interval_size = 1.0 / count;
    double total_area = 0;

    #pragma omp parallel for reduction(+:total_area)
    for(size_t i = 0; i < count; i++)
    {
        double x = (i+0.5) * interval_size;

        double value = 4 / (1 + x * x);
        double area = value * interval_size;

        total_area += area;
    }

    double pi_calculated = total_area;

    timer.stop();



    printf("\n");
    printf("Actual value of PI:     3.1415926535897932...\n");
    printf("\n");
    printf("Calculated value of PI: %.16f\n", pi_calculated);
    printf("\n");
    printf("Execution time: %10.3f ms\n", timer.get_seconds() * 1000);

    return 0;
}
