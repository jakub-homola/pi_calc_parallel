#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib




def func(x):
    return 4 / (1 + x * x)

def my_linspace(left, right, count):
    span = right - left
    vals = list()
    for i in range(0,count):
        vals.append(left + float(i) / (count-1) * span)
    return vals




func_x = my_linspace(0,1,1000)
func_y = [func(x) for x in func_x]



my_figsize = (1200/100.0, 800/100.0)
padding_x = 0.1
padding_y = 0.4
num_intervals = 10






plt.figure(figsize=my_figsize)
plt.fill_between(func_x, func_y, color="blue", alpha=0.1)
plt.plot(func_x, func_y, color="blue", linewidth=4)
plt.plot([0,0,1,1], [4,0,0,2], color="blue", linewidth=1)
plt.grid(True)
plt.xlim(0 - padding_x, 1 + padding_x)
plt.ylim(0 - padding_y, 4 + padding_y)
plt.savefig("img_integral.png")
plt.close()



plt.figure(figsize=my_figsize)
# plt.fill_between(func_x, func_y, color="blue", alpha=0.1)
plt.plot(func_x, func_y, color="black", linewidth=4)
# plt.plot([0,0,1,1], [4,0,0,2], color="blue", linewidth=1)
for i in range(num_intervals):
    left = i / num_intervals
    right = (i+1) / num_intervals
    center = (left + right) / 2
    val = func(center)
    top_x = [left,right]
    top_y = [val,val]
    plt.fill_between(top_x, top_y, color="red", alpha=0.1)
    plt.plot(top_x, top_y, color="red", linewidth=4)
    plt.plot([left,left,right,right], [val,0,0,val], color="red", linewidth=1)

plt.grid(True)
plt.xlim(0 - padding_x, 1 + padding_x)
plt.ylim(0 - padding_y, 4 + padding_y)
plt.savefig("img_rectangles.png")
plt.close()
