#!/usr/bin/env python3

import sys
import time


count = 10000000
if len(sys.argv) > 1:
    count = int(sys.argv[1])

print(f"Number of intervals: {count} = {count//1000000} million")
print()
print("Calculating PI sequentially")
print()
print("...")



time_start = time.monotonic()

interval_size = 1.0 / count
total_area = 0

for i in range(count):
    x = (i+0.5) * interval_size

    value = 4 / (1 + x * x)
    area = value * interval_size

    total_area += area

pi_calculated = total_area

time_stop = time.monotonic()
time_ms = (time_stop - time_start) * 1000



print()
print("Actual value of PI:     3.1415926535897932...")
print()
print(f"Calculated value of PI: {pi_calculated:.16f}")
print()
print(f"Execution time: {time_ms:10.3f} ms")
print()
