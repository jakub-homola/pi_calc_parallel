
# pi_calc_parallel

The common function of all the codes is to estimate the mathematical constant $\pi$.

It holds, that

$\pi = \int_0^1 \frac{4}{1+x^2}$

That is, the area under the curve on the below picture is exactly equal to $\pi$.

![calculating PI as and integral](img_integral.png)

We can estimate the area using the following algorithm. Cut the x-axis into several intervals, replace the curve on each interval with a simple line, calculate the area of the created rectangles, and add the areas of all rectangles together, as shown in the picture below.

![calculating PI as a sum of rectangles](img_rectangles.png)

The main point here is, that this algorithm can be easily parallelized. The codes in this repository show how this can be parallelized using multiple parallelization techniques.











---

my notes, please ignore:

ml OpenMPI/4.1.6-NVHPC-24.1-CUDA-12.4.0

export OMP_PLACES=cores

export OMP_PROC_BIND=close

export OMP_NUM_THREADS=16

mpirun -n 4 --map-by ppr:1:numa --bind-to numa ./program.x
