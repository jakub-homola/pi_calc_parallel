#!/usr/bin/env python3

import sys
import time
from mpi4py import MPI


count = 10000000
if len(sys.argv) > 1:
    count = int(sys.argv[1])

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

if mpi_rank == 0:
    print(f"Number of intervals: {count} = {count//1000000} million")
    print()
    print("Calculating PI using mpi4py")
    print()
    print("...")



time_start = time.monotonic()

interval_size = 1.0 / count
total_area_my_process = 0

count_per_process = count // mpi_size
leftover = count % mpi_size
my_i_start = mpi_rank * count_per_process + min(mpi_rank, leftover)
my_i_end = (mpi_rank+1) * count_per_process + min(mpi_rank+1, leftover)

for i in range(my_i_start, my_i_end):
    x = (i+0.5) * interval_size

    value = 4 / (1 + x * x)
    area = value * interval_size

    total_area_my_process += area

total_area = MPI.COMM_WORLD.reduce(total_area_my_process, MPI.SUM, 0)

pi_calculated = total_area

time_stop = time.monotonic()
time_ms = (time_stop - time_start) * 1000



if mpi_rank == 0:
    print()
    print("Actual value of PI:     3.1415926535897932...")
    print()
    print(f"Calculated value of PI: {pi_calculated:.16f}")
    print()
    print(f"Execution time: {time_ms:10.3f} ms")
    print()
