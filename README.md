# Astro 528 Lab 8
## GPU Computing

During class, please either work on exercises with a partner (so your group is only using one GPU) or, if you're working indvidually, log in to Roar via ssh and submit the exercises as batch jobs via the command line.  The pbs scripts will generate outputs that you can view without using a GPU.  If you run the exercises as batch jobs, then feel free to record your responses in separate markdown files, rather than in the notebooks.  

## Exercise 1:  GPU Computing I: Getting Started with GPU Computing & Linear Algebra
### Goals:  
- Run GPU code on Roar
- Accelerate linear algebra computations with GPU 
- Recognize what problem sizes and likely to result in acceleration with a GPU for linear algebra

Work through ex1.jl, either in the Pluto notebook (with a classmate or outside of class) or by submitting ex1.pbs and viewing the output figures and html file.  

## Exercise 2:  GPU Computing II: GPU Kernels, Reductions
### Goals:  
- Perform custom scientific computations using a high-level GPU interface
- Improve performance by reducing kernel launches via broadcasting and GPU kernel fusion
- Improve performance by reducing memory transfers via GPU reductions
- Recognize what types of problems and problem sizes are likely to result in acceleration with a GPU  when using a custom kernel

Work through ex2.ipynb, either in the Jupyter notebook (with a classmate or outside of class) or by submitting ex2.pbs and viewing the output html file.  
