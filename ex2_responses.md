# Astro 528, Fall 2021
## Lab 8, Exercise 2 Responces

If maybe useful to record your responses in this file, if you run ex1.ipynb as batch job, rather than as an interactive notebook.


2a. How much faster was performing the computations on the GPU than on the CPU?



2b.  What do you predict for the compute time for the same GPU kernel, except with the workgroup size set to 1?



2c. How did the performance with a workgrop size of 1 compare to your predictions?



2d.   Before you run the benchmarks, what do you expect for the GPU performance if we use a workgroup size equal to the warpsize?  What if we use a workgroup size of 1?  Explain your reasoning.  



2e.  How did the benchmarking results for the GPU version compare to your expectations?  What could explain the differences? 



2f.  What do you predict for the time required to transfer the input data to the GPU, to compute the predicted velocities (allocating GPU memory to hold the results), to calcualte chi-squared on the GPU and to return just the chi-squared value to the CPU?



2g.  How did the benchmarking results compare to your prediction?  What might explain any differences?



2h. What do you predict for the time required to compute the predicted velocities (using a pre-allocated workspace on the GPU), to compute chi-squared on the GPU and to return just the chi-squared value to the CPU?



2i.  How did the benchmarking results compare to your prediction?  What might explain any differences?



2j. What do you predict for the speed-up factor of the GPU relative to the CPU?



2k.  How did the benchmarking results compare to your prediction?  What might explain any differences?




2l. What do you observe for the speed-up factor of the GPU relative to the CPU now that we're performing the reduction on the GPU with a pre-allocated workspace?



2m.  If you still have some time, try changing `num_models` and rerunning the affected cells.   How many models do you need to evaluate in parallel to get at least a factor of 50 performance improvement over the CPU?


