#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "timer.h"



int main(int argc, char ** argv)
{
    size_t count = 1000000000;
    if(argc > 1) count = atoll(argv[1]);

    printf("Number of intervals: %zu\n", count);
    printf("\n");
    printf("Calculating PI sequentially\n");
    printf("\n");
    printf("...\n");



    Timer timer;
    timer.start();

    double interval_size = 1.0 / count;
    double total_area = 0;

    for(size_t i = 0; i < count; i++)
    {
        double x_left = i * interval_size;
        double x_right = (i+1) * interval_size;
        double x = (x_left + x_right) / 2;

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
