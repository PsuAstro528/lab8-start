# Astro 528, Fall 2023
## Lab 8, Exercise 1 Responces

If maybe useful to record your responses in this file, if you only run ex1.jl as batch job, rather than as an interactive notebook.

1a.  What do you expect for the maximum difference in any element in b_h and b_d?



1b. How did the results compare to your expectations?



1c.  After swtiching to 64-bit floating point arithmetic, how do the results compare to your expectations?  



1d. Looking at the three above histograms, how long does it take to launch the GPU kernel and return flow control to the CPU, without waiting for the GPU tasks to complete? 



1e. Again looking at the last three histograms, how long did it take to complete the calculation and store to to an array on the GPU? How does this compare to the cost of launching the kernel? What are the implications for the ammount of work you'd want per GPU call in order to make efficient use of the GPU?



1f. For what problem sizes does the runtime of the CPU and GPU become comparable for each of the linear algebra exercises considered above?



1g. How large a batch of 128x128 systems does the GPU need before it is faster than solving the same number and size systems on the CPU?  By what factor is the GPU faster (once you have a large enough number of systems)?


1h.  Try reducing the size of the linear systems to 40x40.  Now, how large a batch of systems does the GPU need before it is faster than solving the same number and size systems on the CPU?  By what factor is the GPU faster (once you have a large enough number of systems)?


1i.  Try reducing the size of the linear systems to 10x10.  Now, how does the GPU performance compare to the CPU performance?



1h. Does your project involve a significant ammount of compute time going into linear algebra? If so, would it make sense to use  a GPU for the linear algebra in your project? Explain your reasoning.
