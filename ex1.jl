### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 4be1d24b-1932-4b75-943e-8cc8caf24241
using CUDA

# ╔═╡ 8c46077d-3fb4-499f-bfd0-94faf806eed1
using PlutoUI, PlutoTeachingTools, PlutoTest

# ╔═╡ 701d55a4-6963-4988-a3bf-f76b2556b9e7
using BenchmarkTools, Statistics

# ╔═╡ 1c989a8d-436a-4023-a637-d4ba26b33620
using LinearAlgebra

# ╔═╡ 3e61f670-5113-40f8-b8f4-4982c663eb19
begin
   using Plots
end

# ╔═╡ fb4509a1-e1aa-45ed-96c9-0e6644a5eda1
md"""
# Astro 528, Lab 8, Exercise 1
### GPU Computing I:  Getting Started & Linear Algebra
"""

# ╔═╡ 47d2db32-cfd7-46cc-81de-09f366b61dc9
md"""
In this lab exercise, we'll verify that we have access to a GPU, learn a little about its properties and benchmark some simple operations on a GPU to see how performance compares.  While most modern laptops have a GPU, not all are designed to support general purpose programming, and most laptop GPUs will be so low power that the benchmarking results won't be very informative.  Therefore, students are advised to run the exercises in this lab on the Roar, rather than their own system (unless they're sure that they have setup their own system for GPU computing properly).  
All students registered for Astro 528 should have access to 
a GPU as part of the class allocation.  However, there may be times when all the GPUs reserved for the class are in use (e.g., during class, near deadlines).  

While the ICDS-ACI portal allows users to request a JupyterLab session that includes access to a GPU, if everyone tries to do this at once, then some people will likely be blocked out.  We can prevent this during class if students either:
- team up with a partner and share one JupyterLab session, or
- view the html version of [this notebook](https://psuastro528.github.io/lab8-start/ex1.html) online rather than running it via the JupyterLab server.  

In either case, you'll modify the ex1.pbs script as necessary and use `qsub` to submit it as a batch job that will run this script and save the resulting figures to disk.  
If you access a GPU via JupyterLab session at another time, then please be curtious and close GPU sessions promptly after you're done, so that others can access the GPUs.

As usual, there are some questions for you to think about as you work through the exercise.  You're welcome to put your responses to questions in *either* this notebook or the ex1_responses.md, whichever is more convenient for you.
"""

# ╔═╡ 1b412736-761f-4564-bafa-7d726629a934
md"""# Properties of your GPU and setup

First, we'll load the CUDA package (for accessing NVIDIA GPUs) and check that you have access to at least one GPU."""

# ╔═╡ ee8683e2-d9bc-451d-804e-38d514cb4369
CUDA.devices()

# ╔═╡ 552232bf-f995-429a-a58b-add54ea77b91
md"If we wanted to check whether our current GPU has some specific capability, then it would be useful to look up it's *compute capability*."

# ╔═╡ 7260f7cb-37be-468a-a48a-2762049acb64
[CUDA.capability(dev) for dev in CUDA.devices()]

# ╔═╡ bb9ba9b8-cec6-40c6-850f-8c380b2bb8b9
md"We can also query specific device attributes."

# ╔═╡ 89428d6e-fe54-454c-a8cd-34c149575e58
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

# ╔═╡ 2ef25884-dc3c-4c18-b5f0-daeaf68260b1
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

# ╔═╡ f682634c-1161-49b3-ac90-edb8804de13d
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X), 
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)

# ╔═╡ fb1642c0-ecf7-45e0-86e7-e0190853e2bf
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)

# ╔═╡ d5c676b5-97d6-4e2c-8db5-ae3b583aa096
CUDA.attribute(first(devices()), CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

# ╔═╡ 79a38f78-8e10-4b73-988c-0dbc7a398427
md"Total GPU RAM (in GB): $((totalmem(first(devices())))/1024^3)"

# ╔═╡ ac8d9001-69e3-45f7-80ae-6d7e00341c4d
md"# Linear Algebra"

# ╔═╡ 2b31f16e-48b1-4941-ba72-bbb0464c19cc


# ╔═╡ 225154c3-c3f9-44ef-ab8a-ad00073ff181
md"""

First, we'll test and benchmark some basic linear algebra tasks.  Because linear algebra is so common, there are efficient libraries for baslic linear algebra tasks ([BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)), solving linear systems ([cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html)), and even sparse linear algebra ([cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)) avaliable for GPUs.  Julia's rich type interface allows users to access many of the most common functions by writing the same linear algebra expressions as you would write for vectors and matrices on the CPU, but passing vectors and matrices stored on the GPU.  (There are also less commonly called library functions which do not have julia wrappers yet, but can be called explicitly.)    
"""

# ╔═╡ 6cb46d4d-6e3b-4bb7-b1b5-e82e7294427f
md"Now, let's create some matrices and arrays on the CPU that we'll use for testing purposes."  

# ╔═╡ c739f53b-c7c8-4b49-9ae4-cf8edc246c25
begin
	N = 4*1024
	M = 4*1024
end;

# ╔═╡ 0afda366-146c-408b-9d14-0ce510e2db2c
begin
	A_h = randn(N,M)
	x_h = randn(M);

	b_h = A_h*x_h;
end;

# ╔═╡ 11c4ecad-8fa7-428d-aa73-64c1238c7947
md"""
To perform calculations on a GPU, we'll need to send the data from the CPU to the GPU.  The CUDA pacakge provides a datatype `CuArray` for storing arrays in the GPU or *device* memory.  Manually specifying when to send data back and forth does provide some extra options that be good for efficiency, but it also requires more care.  So instead, we use `CuArray`'s that take care of moving the data between the CPU (or "host") memory system and the GPU (or "device") memory system for us.  

We can create a 'CuArray' from an existing Array simply with the `cu(.)` function.  Then, we can proceed to do arithmetic on them with the same syntax as when using standard Arrays.  (Generic programming is amazing!)
"""

# ╔═╡ 378488d1-c86c-4db0-9416-739896ac6632
begin
	A_d = cu(A_h)
	x_d = cu(x_h)
	b_d = A_d*x_d
end;

# ╔═╡ 81056fe9-34e4-4f1f-850a-2b10df79cd4a
md"First, let's check the type of each variable."

# ╔═╡ f82bbb2d-71d3-43af-a949-3fb9b38ee0cd
typeof(A_d), typeof(x_d), typeof(b_d)

# ╔═╡ 10925beb-c6b1-47e1-9003-279fadc465a0
warning_box(md"""
Note that the `cu` function decided to convert the data from `Float64` to `Float32`.  This can provide a non-trivial speed-up for GPU calculations, but also increases the risk that floating-point error may lead to unacceptablely inaccurate results.  
We'll want to compare the accuracy of the calculations below.""")

# ╔═╡ 8b486f0f-f633-4e49-8ef6-341644a306f2


# ╔═╡ 2fec476d-0190-4394-92b9-b52fda23dd87
md"""
By using two CuArray's as input, the result of the calculation is also stored as a CuArray and left on the GPU.  While it is technically possible to access individual elements, it is quite slow.  So when we want to be able to access a whole array, then it will be faster to copy all the data at once to CPU.  We can bring the full result back to the host, using `collect(.)`  Let's compare the two below.
"""

# ╔═╡ bf804c53-9895-431c-a1be-60e21b7f9595
with_terminal() do 
	@time CUDA.@allowscalar b_d[1]
	@time CUDA.@allowscalar b_d[1]
end

# ╔═╡ b40e67e0-0b14-45b1-87c1-1a94a536b84d
with_terminal() do
	@time b_comp_h = collect(b_d)
	@time b_comp_h = collect(b_d)
end

# ╔═╡ fa37c38e-8658-440f-9f09-4a8fca7bac82
md"""
Next, we'll compare the results.  

1a.  What do you expect for the maximum difference in any element in `b_h` and `b_d`?
"""

# ╔═╡ 75198535-faef-4e9a-920e-5923a26f5239
response_1a =  missing # md"Insert your response either here or in ex1_responses.md"

# ╔═╡ b7b0b025-ee38-47f8-bdc6-a17e2f814969
maximum(abs.(collect(b_d) .- b_h))

# ╔═╡ 21beea5f-d229-434c-a11e-5d2caec52dc7
md"""
1b. How did the results compare to your expectations?  
"""

# ╔═╡ 223c81b2-5e2c-483a-9a60-f77524f76936
response_1b = missing  # md"Insert your response either here or in ex1_responses.md"

# ╔═╡ e5b93a70-2804-429b-95b8-487526aaa62f
md"""
One thing to keep in mind is that most "consumer grade" GPUs are designed to only perform single-precision arithmetic.  Even some GPUs that do support double precission are often significantly slower at double precission arithmetic than single precission.  The difference is particularly noticable for consumer grade GPUs and some recent GPUs optimized for machine learning tasks.  When we want to upload data to the GPU in double precission, we can specify that explicitly.  Below we will reupload our data to the GPU, but this time using double precision on the GPU, so we can test its accuracy.
"""

# ╔═╡ 280a9ce0-7fc0-4953-a450-30908620793a
begin
	A_d64 = CuArray{Float64}(A_h)
	x_d64 = CuArray{Float64}(x_h)
	b_d64 = A_d64*x_d64
	maximum(abs.(collect(b_d64) .- b_h))
end

# ╔═╡ da6782ac-daf0-425f-bf1f-96d4e99caade
md"""
1c.  After swtiching to 64-bit floating point arithmetic, how do the results compare to your expectations?  
"""

# ╔═╡ 19c48df4-e831-458a-993e-ac6bbd3b2cf5
response_1c = missing

# ╔═╡ aaa959b6-3eec-47f2-8f67-3308c6279edd
md"""
## Benchmarking GPU for Linear Algebra
Now, we'll do some benchmarking of the CPU vs GPU for basic linear algebra.

Note that we'll be able to use the exact same macros to benchmark code running on either the CPU or the GPU.  To keep things reasonably fast, we'll specify that we only want Julia to benchmark each calculation a few times.  
"""

# ╔═╡ f7f24f2f-a085-49a4-a264-2a55159bf00c
@benchmark b_h_comp = $A_h*$x_h seconds=1

# ╔═╡ 9cf8fdc6-16ba-405c-a9ce-2f22162fa8b8
@benchmark b_d = $A_d*$x_d seconds=1

# ╔═╡ 4a9161e9-57f8-43b4-84bc-3b7d8807cbb5
md"""
The histogram of the time required on the GPU is likely bimodal.  What could cause that?  Remember, that calls to the GPU run asynchronously.  Sometimes, we timed how long to start the calculation on the GPU and return control to the CPU.  Othertimes, the GPU kernel couldn't start right away because it was still performing work for the last kernel call, so we had to wait for the last kernel to finish before starting the next one.  (It's now possible to launch kernels that use only a fraction of the GPU's multiprocessors, so that multiple kernels are running at once.  But by default, Julia is trying to use all the multiprocessors on the GPU for each calculation.) 
If we want to benchmark how long is required to the calculation to complete, we need to tell Julia to wait until the calculation has been completed with the `CUDA.@sync` macro.  (This can also be very useful for and making sure that there is no risk of variables becoming unsynchronized and overwritten in an order different than we intended.)
"""

# ╔═╡ e3b5fd7c-7e5c-4e68-84f4-db5a66b72db1
@benchmark CUDA.@sync( b_d = $A_d*$x_d) seconds=1

# ╔═╡ 203ba323-9b71-49dd-956b-de677cc8a033
md"1d. Looking at the three above histograms, how long does it take to launch the GPU kernel and return flow control to the CPU, without waiting for the GPU tasks to complete?  "

# ╔═╡ 2af9ae78-b042-490e-ad79-e0c87dcb2eb3
response_1d = missing # md"Insert your response either here or in ex1_responses.md"

# ╔═╡ 32604896-2034-42f7-bc49-6117fe13f6d2
md"""
1e.  Again looking at the last three histograms, how long did it take to complete the calculation and store to to an array on the GPU?  How does this compare to the cost of launching the kernel?  What are the implications for the ammount of work you'd want per GPU call in order to make efficient use of the GPU?
"""

# ╔═╡ ee1f72d7-aad1-4fb7-b49c-09f6bd32ab51
response_1e = missing # md"Insert your response either here or in ex1_responses.md"

# ╔═╡ 4977a149-2eeb-4f10-8d94-111dcd85b59f
md"## Matrix solve"

# ╔═╡ c5b47249-a687-4cbd-ab25-37264a4dd467
md"Here, we'll evaluate the speed-up factors for solving a linear system on the GPU, using 32 and 64-bit floating point arithmetic."

# ╔═╡ 874f2510-62d0-41e1-80fe-3a60846dcbab
begin
	solve_for_x_h = A_h \ b_h
	walltime_lu_solve_h = @elapsed solve_for_x_h = A_h \ b_h
end

# ╔═╡ 47e1977e-6640-4823-b32a-1ca714220351
begin
	CUDA.@sync (solve_for_x_d = A_d \ b_d);
	walltime_lu_solve_d = @elapsed (CUDA.@sync solve_for_x_d = A_d \ b_d)
	walltime_lu_solve_h / walltime_lu_solve_d
end

# ╔═╡ 7a83aa4e-ca54-45a0-8547-402537053892
begin
	CUDA.@sync (solve_for_x_d64 = A_d64 \ b_d64);
	walltime_lu_solve_d64 = @elapsed (CUDA.@sync solve_for_x_d64 = A_d64 \ b_d64)
	walltime_lu_solve_h / walltime_lu_solve_d64
end

# ╔═╡ 24f9b39f-e99f-419b-afb5-61c9a1385915
md"""
Let's check to make sure that the GPU results are accurate.
"""

# ╔═╡ 062f9dfc-49d7-4894-92d2-52a0fb4aa906
@test maximum(abs.(A_h \ b_h .- collect(A_d \ b_d))) < 1e-2

# ╔═╡ bd2593c0-d647-4807-a904-716801a40703
@test maximum(abs.(A_h \ b_h .- collect(A_d64 \ b_d64))) < 1e-8

# ╔═╡ c1f08951-4303-4599-ac6c-1884c46dc5c7
md"## Benchmarking vs problem size"

# ╔═╡ 45f9f228-dfca-4f37-b44e-45dc94cb83c3
md"""
As before, the relative efficiency of the CPU and GPU depend on the problem size.  Below, we'll benchmark matrix-vector multiply, matrix-matrix multiply, and solving a linear system as a function of problem size.  
"""

# ╔═╡ ebe44e7a-fe7c-4ff3-897a-f7c1d7fba2b0
tip(md"""
In case you're running this as a batch job rather than in an interactive notebook, the figures of benchmarking results will be saved to '*.png' files, so you can inspect them after your jobs finish.
""")

# ╔═╡ a54a1ceb-e741-44b5-b0cd-ecda8c457a19
n_plot = [2^i for i in 4:12]

# ╔═╡ 99885312-83a9-48e5-a647-1692fc6a9b89
m_plot = n_plot

# ╔═╡ 17652979-ef2c-4b13-8c3b-102f01a1f312
md"""
1f.  For what size linear algebra operations, does the runtime of the CPU and GPU become comparable?  
"""

# ╔═╡ c14f21e0-f4d0-42d7-bdd2-6953334d28f3


# ╔═╡ 8f0cb976-0734-47f4-951e-119288f7506c
response_1f = missing # md"Insert your response either here or in ex1_responses.md"

# ╔═╡ e056c322-139e-4ab5-819d-cf804f48902c
md"""
1g. Does your project involve a significant ammount of time performing linear algebra? If so, are there a few very large matrix operations?  Are there many small matrix opertaions?  Would it make sense to use  a GPU for the linear algebra in your project? Explain your reasoning.  (Feel free to create your own benchmarks based on an operation more similar to what is needed for your project.)
"""

# ╔═╡ 56486b73-112d-46e1-91f8-0981e0a49df3
response_1g =  missing # md"Insert your response either here or in ex1_responses.md"

# ╔═╡ 2722f671-9f2d-40ec-8a67-8032998a41bc
md"# Helper Code"

# ╔═╡ c1d3f9f3-269f-43b4-9336-b376b7e98361
function cpu_benchmark_mul_mat_vec(n,m)
	A = rand(n,m)
	x = rand(m)
	@elapsed b = A*x
end

# ╔═╡ a4624650-de76-4a91-b261-6d73525eb2eb
function gpu_benchmark_mul_mat_vec(n,m; eltype=Float64)
	A_d = CUDA.rand(eltype,n,m)
	x_d = CUDA.rand(eltype,m)
	b_d = A_d*x_d
	b_h = collect(b_d)

	time_init = @elapsed CUDA.@sync begin
		A_d = CUDA.rand(eltype,n,m)
		x_d = CUDA.rand(eltype,m)
	end
	time_execute = @elapsed CUDA.@sync begin		
		b_d = A_d*x_d
    end
	time_download = @elapsed CUDA.@sync begin
		b_h = collect(b_d)
	end
	time_total = time_init + time_execute + time_download
	return (; time_total, time_init, time_execute, time_download)
end


# ╔═╡ f6411722-faaa-463f-af5a-fd0e820a196c
begin
	gpu_mul_mat_vec_64_results = map((n,m)->gpu_benchmark_mul_mat_vec(n,m), n_plot, m_plot)
	gpu_mul_mat_vec_64_total = map(x->x.time_total,gpu_mul_mat_vec_64_results)
	gpu_mul_mat_vec_64_exec = map(x->x.time_execute,gpu_mul_mat_vec_64_results)
	gpu_mul_mat_vec_32_results = map((n,m)->gpu_benchmark_mul_mat_vec(n,m,eltype=Float32), n_plot,m_plot )
	gpu_mul_mat_vec_32_total = map(x->x.time_total,gpu_mul_mat_vec_32_results)
	gpu_mul_mat_vec_32_exec = map(x->x.time_execute,gpu_mul_mat_vec_32_results)
end;

# ╔═╡ 99262deb-6206-498f-bb94-a779578ffeab
begin
	cpu_mul_mat_vec_times_total = map(n->cpu_benchmark_mul_mat_vec(n,n), n_plot )
end;

# ╔═╡ eea95112-19e5-4371-a246-9941eab16cc2
let
	plt = plot(xscale=:log10, yscale=:log10, legend=:topleft);
	scatter!(plt, n_plot, cpu_mul_mat_vec_times_total, color=2, label="CPU (total)");
	plot!(plt, n_plot, cpu_mul_mat_vec_times_total, color=2, label="CPU (total)");
	scatter!(plt, n_plot, gpu_mul_mat_vec_64_total, color=1, label="GPU (64bit, total)");
	plot!(plt, n_plot, gpu_mul_mat_vec_64_exec, color=1, label="GPU (64bit, execute)");
	scatter!(plt, n_plot, gpu_mul_mat_vec_32_total, color=3, label="GPU (32bit, total)");
	plot!(plt, n_plot, gpu_mul_mat_vec_32_exec, color=3, label="GPU (32bit, execute)");
	xlabel!(plt, "N: Number of rows in matrix");
	ylabel!(plt, "Wall time");
	title!(plt,"CPU vs GPU Performance:\nMatrix (n×m)-Vector (m) Multiply");
	savefig("benchmarks_mul_mat_vec.png");
	plt
end

# ╔═╡ 1939ed6c-1238-4147-ae90-caae629452ce
function cpu_benchmark_mul_mat_mat(n,m,p)
	A = rand(n,m)
	x = rand(m,p)
	@elapsed b = A*x
end

# ╔═╡ ce31372c-4686-4896-bb76-f273777bf758
function gpu_benchmark_mul_mat_mat(n,m,p; eltype=Float64)
	A_d = CUDA.rand(eltype,n,m)
	x_d = CUDA.rand(eltype,m,p)
	b_d = A_d*x_d
	b_h = collect(b_d)

	time_init = @elapsed CUDA.@sync begin
		A_d = CUDA.rand(eltype,n,m)
		x_d = CUDA.rand(eltype,m,p)
	end
	time_execute = @elapsed CUDA.@sync begin		
		b_d = A_d*x_d
    end
	time_download = @elapsed CUDA.@sync begin
		b_h = collect(b_d)
	end
	time_total = time_init + time_execute + time_download
	return (; time_total, time_init, time_execute, time_download)
end


# ╔═╡ ffa851f2-a234-4ee1-85ee-4c611e7c1385
begin
	gpu_mul_mat_mat_64_results = map((n,m,p)->gpu_benchmark_mul_mat_mat(n,m ,p), n_plot, m_plot,  n_plot)
	gpu_mul_mat_mat_64_total = map(x->x.time_total,gpu_mul_mat_mat_64_results)
	gpu_mul_mat_mat_64_exec = map(x->x.time_execute,gpu_mul_mat_mat_64_results)
	gpu_mul_mat_mat_32_results = map((n,m,p)->gpu_benchmark_mul_mat_mat(n,m,p,eltype=Float32), n_plot,m_plot,n_plot )
	gpu_mul_mat_mat_32_total = map(x->x.time_total,gpu_mul_mat_mat_32_results)
	gpu_mul_mat_mat_32_exec = map(x->x.time_execute,gpu_mul_mat_mat_32_results)
end;

# ╔═╡ 16016799-3ac3-471c-add6-1f38e0fe9749
begin
	cpu_mul_mat_mat_times_total = map(n->cpu_benchmark_mul_mat_mat(n,n,n), n_plot )
end;

# ╔═╡ ea1c8c71-3e35-4fae-b8ed-70a099561090
let
	plt = plot(xscale=:log10, yscale=:log10, legend=:topleft);
	scatter!(plt, n_plot, cpu_mul_mat_mat_times_total, color=2, label="CPU (total)");
	plot!(plt, n_plot, cpu_mul_mat_mat_times_total, color=2, label="CPU (total)");
	scatter!(plt, n_plot, gpu_mul_mat_mat_64_total, color=1, label="GPU (64bit, total)");
	plot!(plt, n_plot, gpu_mul_mat_mat_64_exec, color=1, label="GPU (64bit, execute)");
	scatter!(plt, n_plot, gpu_mul_mat_mat_32_total, color=3, label="GPU (32bit, total)");
	plot!(plt, n_plot, gpu_mul_mat_mat_32_exec, color=3, label="GPU (32bit, execute)");
	xlabel!(plt, "N: Number of rows in first matrix");
	ylabel!(plt, "Wall time");
	title!(plt,"CPU vs GPU Performance:\nMatrix (n×m)-Matrix (m×n) Multiply");
	savefig("benchmarks_mul_mat_mat.png");
	plt
end

# ╔═╡ 941d5524-11ac-436e-8526-563ef2ad28b6
function cpu_benchmark_lu_solve(n,m)
	A = rand(n,m)
	b = rand(m)
	@elapsed x = A\b
end

# ╔═╡ cd3d134c-8175-443b-8fdd-02dbca473159
function gpu_benchmark_lu_solve(n,m; eltype=Float64)
	A_d = CUDA.rand(eltype,n,m)
	b_d = CUDA.rand(eltype,m)
	x_d = A_d\b_d
	x_h = collect(x_d)

	time_init = @elapsed CUDA.@sync begin
		A_d = CUDA.rand(eltype,n,m)
		b_d = CUDA.rand(eltype,m)
	end
	time_execute = @elapsed CUDA.@sync begin		
		x_d = A_d\b_d
    end
	time_download = @elapsed CUDA.@sync begin
		x_h = collect(x_d)
	end
	time_total = time_init + time_execute + time_download
	return (; time_total, time_init, time_execute, time_download)
end


# ╔═╡ b2767029-86b5-4b62-b1fb-a0c27c7e118d
begin
	gpu_solve_64_results = map(n->gpu_benchmark_lu_solve(n,n), n_plot)
	gpu_solve_64_total = map(x->x.time_total,gpu_solve_64_results)
	gpu_solve_64_exec = map(x->x.time_execute,gpu_solve_64_results)
	gpu_solve_32_results = map(n->gpu_benchmark_lu_solve(n,n,eltype=Float32), n_plot )
	gpu_solve_32_total = map(x->x.time_total,gpu_solve_32_results)
	gpu_solve_32_exec = map(x->x.time_execute,gpu_solve_32_results)
end;

# ╔═╡ 75fefe3f-2170-4231-a676-f2b8aef49100
begin
	cpu_solve_times_total = map(n->cpu_benchmark_lu_solve(n,n), n_plot[1:end-1] )
end;

# ╔═╡ 2d3ae427-1418-48b7-aa29-754fecbd9283
let
	plt = plot(xscale=:log10, yscale=:log10, legend=:topleft);
	scatter!(plt, n_plot[1:end-1], cpu_solve_times_total, color=2, label="CPU (total)");
	plot!(plt, n_plot[1:end-1], cpu_solve_times_total, color=2, label="CPU (total)");
	scatter!(plt, n_plot, gpu_solve_64_total, color=1, label="GPU (64bit, total)");
	plot!(plt, n_plot, gpu_solve_64_exec, color=1, label="GPU (64bit, execute)");
	scatter!(plt, n_plot, gpu_solve_32_total, color=3, label="GPU (32bit, total)");
	plot!(plt, n_plot, gpu_solve_32_exec, color=3, label="GPU (32bit, execute)");
	xlabel!(plt, "N: Number of rows in matrix");
	ylabel!(plt, "Wall time");
	title!(plt,"CPU vs GPU Performance:\nMatrix (n×m) Solve");
	savefig("benchmarks_solve.png");
	plt
end

# ╔═╡ eaef6efb-d614-4b54-8599-e5173ac591ab
md"### Example of batched linear algebra operations on GPU"

# ╔═╡ 019f57eb-ee5f-4500-958d-830186d62814
md"""
I didn't integrate the following example into the main exercise.  But if anyone is interested in how to perform many small linear algebra operations on the GPU, here's an example.  To find other batched functions or to make sense of the outputs, you'll want to look at the documentation for [CUBLAS](https://docs.nvidia.com/cuda/cublas/index.html), the library of CUDA functions that provide these operations.  For some problem sizes, the CUBLAS functions are very efficient.  For other problem sizes, you can work more efficiently using multiple asyncrhonous kernel calls or just using multi-threading on the CPU. 
"""

# ╔═╡ 80a04a89-b23e-4d2f-9a4d-ce887ba0cf61
md"""
### Batched Linear Algebra Operations via CUBLAS 
"""

# ╔═╡ bf17466e-b5a3-4180-acfb-362c5da55d70
num_systems = 500

# ╔═╡ 185cdee9-84e7-443f-a6a3-7ae6369e956e
n_batched_matrices = 128

# ╔═╡ 73b4d226-4062-44d7-9f23-b16864b93b31
"""
`make_inputs_batched_Ax_eq_b(n, num_systems; elty)`

Inputs:
- `n`: Solve nxn matrices
- `num_systems`: Number of systems to solve
- `elty`: type of arrays to solve (Float64)
Outputs:

NamedTuple with arrays containing `num_systems` values for each of `A`, `x` and `b` where each `A x = b` 
"""
function make_inputs_batched_Ax_eq_b(n::Integer, num_systems::Integer; elty::Type = Float64)
	k = 1               # CUDA uses a nx1 matrix instead of a vector	
    A = [ begin
			R = rand(elty,n,n)
			Q = R'*R
			(Q'+Q)/2
		end for i in 1:num_systems]
	x = [ rand(elty,n,k) for i in 1:num_systems]
	b = [ A[i]*x[i] for i in 1:num_systems]
	return (;A, x, b)
end

# ╔═╡ 8c727358-0695-43ae-96b2-ded282e0dedf
"""
`copy_array_of_arrays_to_gpu(A; force_sync)`

Input:
- `A`:  Array of 2-darrays of numbers, all stored on the host
- `force_sync`: Synchronize before returning (false)
Output:
- CuArray of 2-d CuArrays of data, all stored on the GPU
"""
function copy_array_of_arrays_to_gpu(A::TArrayOuter; force_sync::Bool = false) where { TVal<:Number, TArrayInner<: AbstractArray{TVal}, TArrayOuter<: AbstractArray{TArrayInner,1} }
	elty = TVal
	# Allocate device arrays of matrices 
	d_A = CuArray{elty, 2}[]
    # Move each array to GPU
	for i in 1:length(A)
       push!(d_A,CuArray(A[i]))
	end
	if force_sync
		CUDA.@sync d_A
	end
	return d_A
end

# ╔═╡ 1a5e09f6-0b83-4e13-af89-4364e219ff16
begin
	A, x, b = make_inputs_batched_Ax_eq_b(n_batched_matrices,num_systems)
	d_A = copy_array_of_arrays_to_gpu(A)
	d_b = copy_array_of_arrays_to_gpu(b)
	output_qr_d, output_x_d, output_flag_d = CUBLAS.gels_batched('N', d_A, d_b) 
end

# ╔═╡ fe609884-c598-4391-ab06-4ba7166b65ff
# Test solve suceeded
CUDA.@sync output_flag_d; @test all(collect($output_flag_d).==0)

# ╔═╡ f6119a25-00e6-4a65-a3d8-8a56e8719fb5
# Test that results are close to the correct solution
CUDA.@sync output_x_d;  @test all([all(isapprox.(x[i], collect($output_x_d[i]), rtol=1e-2)) for i in 1:num_systems])

# ╔═╡ f1233e72-adc0-43dc-8796-055cf9baf4e7
let 
	CUDA.@sync output_x_d
	# Check absolute value of worst element of solution
	maximum(map(i->maximum(abs.(x[i].-collect(output_x_d[i]))), 1:num_systems))
end

# ╔═╡ 370b5a64-9fcb-4431-99c1-d0dee3f64932
function gpu_benchmark_batch_solve(n_matrix::Integer, num_systems::Integer;
										elty::Type = Float64)
	A, x, b = make_inputs_batched_Ax_eq_b(n_matrix,num_systems; elty)
	time_init = @elapsed CUDA.@sync begin
		d_A = copy_array_of_arrays_to_gpu(A)
		d_b = copy_array_of_arrays_to_gpu(b)
	end 
	time_execute = @elapsed CUDA.@sync begin 
		output_qr_d, output_x_d, output_flag_d = CUBLAS.gels_batched('N', d_A, d_b)
	end
	time_download =  @elapsed ( CUDA.@sync output_x = collect(output_x_d) )
	time_total = time_init + time_execute + time_download
	return (;output_x, time_total, time_init, time_execute, time_download)
end

# ╔═╡ 21c26eca-5896-4a67-801a-86a228edc2c5
function cpu_benchmark_batch_solve(n_matrix::Integer, num_systems::Integer;
										elty::Type = Float64)
	A, x, b = make_inputs_batched_Ax_eq_b(n_matrix,num_systems; elty)
	local output_x 
	time = @elapsed begin
		output_x = fill(elty[],num_systems)
		for i in 1:length(A)
			output_x[i] = A[i] \ view(b[i],:,1)
 		end
	end
	return time
end

# ╔═╡ a20538f8-1edb-44dc-bee9-989dc6926ee9
begin
	gpu_solve_batch_64_results = map(nsys->gpu_benchmark_batch_solve(n_batched_matrices,nsys, elty=Float64), n_plot)
	gpu_solve_batch_64_total = map(x->x.time_total,gpu_solve_batch_64_results)
	gpu_solve_batch_64_exec = map(x->x.time_execute,gpu_solve_batch_64_results)
	gpu_solve_batch_32_results = map(nsys->gpu_benchmark_batch_solve(n_batched_matrices,nsys, elty=Float32), n_plot)
	gpu_solve_batch_32_total = map(x->x.time_total,gpu_solve_batch_32_results)
	gpu_solve_batch_32_exec = map(x->x.time_execute,gpu_solve_batch_32_results)
end;

# ╔═╡ 7d5c48ff-5806-4bb2-acdd-98a508627bac
cpu_solve_batch_total = map(nsys->cpu_benchmark_batch_solve(n_batched_matrices,nsys, elty=Float64), n_plot);

# ╔═╡ ff118324-f687-44d6-aca9-dc84802767e2
let
	plt = plot(xscale=:log10, yscale=:log10, legend=:topleft);
	scatter!(plt, n_plot, cpu_solve_batch_total, color=2, label="CPU (total)");
	plot!(plt, n_plot, cpu_solve_batch_total, color=2, label="CPU (total)");
	scatter!(plt, n_plot, gpu_solve_batch_64_total, color=1, label="GPU (64bit, total)");
	plot!(plt, n_plot, gpu_solve_batch_64_exec, color=1, label="GPU (64bit, execute)");
	scatter!(plt, n_plot, gpu_solve_batch_32_total, color=3, label="GPU (32bit, total)");
	plot!(plt, n_plot, gpu_solve_batch_32_exec, color=3, label="GPU (32bit, execute)");
	xlabel!(plt, "N: Number of systems to solve");
	ylabel!(plt, "Wall time");
	title!(plt,"CPU vs GPU Performance:\nBatch of Matrix ($n_batched_matrices×$n_batched_matrices) Solves");
	savefig("benchmarks_batch_solve.png");
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoTest = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.2.0"
CUDA = "~3.5.0"
Plots = "~1.22.6"
PlutoTeachingTools = "~0.1.4"
PlutoTest = "~0.1.2"
PlutoUI = "~0.7.16"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.2"
manifest_format = "2.0"
project_hash = "4da966efaf9c95204e7b0599073cb379c44465e5"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "91bd53c39b9cbfb5ef4b015e8b582d344532bd0a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.0"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "2c8329f16addffd09e6ca84c556e2185a4933c64"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.5.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "a1296f0fe01a4c3f9bf0dc2934efbf4416f5db31"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "8a62af3e248a8c4bad6b32cbbe663ae02275e32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "647a54f196b5ffb7c3bc2fec5c9a57fa273354cc"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.13.14"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "bb198ff907228523f3dee1070ceee63b9359b6ab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "81dc6aefcbe7421bd62cb6ca0e700779330acff8"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.25"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f044a2796a9e18e0531b9b3072b0019a61f264bc"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.17.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "070e4b5b65827f82c16ae0916376cb47377aa1b5"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.18+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "60168780555f3e663c536500aa790b6368adc02a"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.3.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a12e56c72edee3ce6b96667745e6cbbe5498f200"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "2e73fe17cac3c62ad1aebe70d44c963c3cfdc3e3"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "716e24b21538abc91f6205fd1d8363f39b442851"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "e7523dd03eb3aaac09f743c23c1a553a8c834416"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.7"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "67c917d383c783aeadd25babad6625b834294b30"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.1.7"

[[deps.PlutoTest]]
deps = ["HypertextLiteral", "InteractiveUtils", "Markdown", "Test"]
git-tree-sha1 = "b7da10d62c1ffebd37d4af8d93ee0003e9248452"
uuid = "cb4044da-4d16-4ffa-a6a3-8cad7f73ebdc"
version = "0.1.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e47cd150dbe0443c3a3651bc5b9cbd5576ab75b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.52"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "609c26951d80551620241c3d7090c71a73da75ab"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.5.6"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6da46b16e6bca4abe1b6c6fa40b94beb0c87f4ac"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.8.8"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore"]
git-tree-sha1 = "0adf069a2a490c47273727e029371b31d44b72b2"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.6.5"
weakdeps = ["Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "a1f34829d5ac0ef499f6d84428bd6b4c71f02ead"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "b7a5e99f24892b6824a954199a45e9ffcc1c70f0"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "24b81b59bd35b3c42ab84fa589086e19be919916"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.11.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─fb4509a1-e1aa-45ed-96c9-0e6644a5eda1
# ╟─47d2db32-cfd7-46cc-81de-09f366b61dc9
# ╟─1b412736-761f-4564-bafa-7d726629a934
# ╠═4be1d24b-1932-4b75-943e-8cc8caf24241
# ╠═ee8683e2-d9bc-451d-804e-38d514cb4369
# ╟─552232bf-f995-429a-a58b-add54ea77b91
# ╠═7260f7cb-37be-468a-a48a-2762049acb64
# ╟─bb9ba9b8-cec6-40c6-850f-8c380b2bb8b9
# ╠═89428d6e-fe54-454c-a8cd-34c149575e58
# ╠═2ef25884-dc3c-4c18-b5f0-daeaf68260b1
# ╠═f682634c-1161-49b3-ac90-edb8804de13d
# ╠═fb1642c0-ecf7-45e0-86e7-e0190853e2bf
# ╠═d5c676b5-97d6-4e2c-8db5-ae3b583aa096
# ╠═79a38f78-8e10-4b73-988c-0dbc7a398427
# ╟─ac8d9001-69e3-45f7-80ae-6d7e00341c4d
# ╠═2b31f16e-48b1-4941-ba72-bbb0464c19cc
# ╟─225154c3-c3f9-44ef-ab8a-ad00073ff181
# ╟─6cb46d4d-6e3b-4bb7-b1b5-e82e7294427f
# ╠═c739f53b-c7c8-4b49-9ae4-cf8edc246c25
# ╠═0afda366-146c-408b-9d14-0ce510e2db2c
# ╟─11c4ecad-8fa7-428d-aa73-64c1238c7947
# ╠═378488d1-c86c-4db0-9416-739896ac6632
# ╟─81056fe9-34e4-4f1f-850a-2b10df79cd4a
# ╠═f82bbb2d-71d3-43af-a949-3fb9b38ee0cd
# ╟─10925beb-c6b1-47e1-9003-279fadc465a0
# ╠═8b486f0f-f633-4e49-8ef6-341644a306f2
# ╟─2fec476d-0190-4394-92b9-b52fda23dd87
# ╠═bf804c53-9895-431c-a1be-60e21b7f9595
# ╠═b40e67e0-0b14-45b1-87c1-1a94a536b84d
# ╟─fa37c38e-8658-440f-9f09-4a8fca7bac82
# ╠═75198535-faef-4e9a-920e-5923a26f5239
# ╠═b7b0b025-ee38-47f8-bdc6-a17e2f814969
# ╟─21beea5f-d229-434c-a11e-5d2caec52dc7
# ╠═223c81b2-5e2c-483a-9a60-f77524f76936
# ╟─e5b93a70-2804-429b-95b8-487526aaa62f
# ╠═280a9ce0-7fc0-4953-a450-30908620793a
# ╟─da6782ac-daf0-425f-bf1f-96d4e99caade
# ╠═19c48df4-e831-458a-993e-ac6bbd3b2cf5
# ╟─aaa959b6-3eec-47f2-8f67-3308c6279edd
# ╠═f7f24f2f-a085-49a4-a264-2a55159bf00c
# ╠═9cf8fdc6-16ba-405c-a9ce-2f22162fa8b8
# ╟─4a9161e9-57f8-43b4-84bc-3b7d8807cbb5
# ╠═e3b5fd7c-7e5c-4e68-84f4-db5a66b72db1
# ╟─203ba323-9b71-49dd-956b-de677cc8a033
# ╠═2af9ae78-b042-490e-ad79-e0c87dcb2eb3
# ╟─32604896-2034-42f7-bc49-6117fe13f6d2
# ╠═ee1f72d7-aad1-4fb7-b49c-09f6bd32ab51
# ╟─4977a149-2eeb-4f10-8d94-111dcd85b59f
# ╟─c5b47249-a687-4cbd-ab25-37264a4dd467
# ╠═874f2510-62d0-41e1-80fe-3a60846dcbab
# ╠═47e1977e-6640-4823-b32a-1ca714220351
# ╠═7a83aa4e-ca54-45a0-8547-402537053892
# ╟─24f9b39f-e99f-419b-afb5-61c9a1385915
# ╠═062f9dfc-49d7-4894-92d2-52a0fb4aa906
# ╠═bd2593c0-d647-4807-a904-716801a40703
# ╟─c1f08951-4303-4599-ac6c-1884c46dc5c7
# ╟─45f9f228-dfca-4f37-b44e-45dc94cb83c3
# ╟─ebe44e7a-fe7c-4ff3-897a-f7c1d7fba2b0
# ╠═a54a1ceb-e741-44b5-b0cd-ecda8c457a19
# ╠═99885312-83a9-48e5-a647-1692fc6a9b89
# ╟─eea95112-19e5-4371-a246-9941eab16cc2
# ╟─ea1c8c71-3e35-4fae-b8ed-70a099561090
# ╠═2d3ae427-1418-48b7-aa29-754fecbd9283
# ╟─17652979-ef2c-4b13-8c3b-102f01a1f312
# ╠═c14f21e0-f4d0-42d7-bdd2-6953334d28f3
# ╠═8f0cb976-0734-47f4-951e-119288f7506c
# ╟─e056c322-139e-4ab5-819d-cf804f48902c
# ╠═56486b73-112d-46e1-91f8-0981e0a49df3
# ╟─2722f671-9f2d-40ec-8a67-8032998a41bc
# ╠═8c46077d-3fb4-499f-bfd0-94faf806eed1
# ╠═701d55a4-6963-4988-a3bf-f76b2556b9e7
# ╠═1c989a8d-436a-4023-a637-d4ba26b33620
# ╠═3e61f670-5113-40f8-b8f4-4982c663eb19
# ╟─c1d3f9f3-269f-43b4-9336-b376b7e98361
# ╟─a4624650-de76-4a91-b261-6d73525eb2eb
# ╟─f6411722-faaa-463f-af5a-fd0e820a196c
# ╟─99262deb-6206-498f-bb94-a779578ffeab
# ╠═1939ed6c-1238-4147-ae90-caae629452ce
# ╠═ce31372c-4686-4896-bb76-f273777bf758
# ╟─ffa851f2-a234-4ee1-85ee-4c611e7c1385
# ╟─16016799-3ac3-471c-add6-1f38e0fe9749
# ╠═941d5524-11ac-436e-8526-563ef2ad28b6
# ╟─cd3d134c-8175-443b-8fdd-02dbca473159
# ╟─b2767029-86b5-4b62-b1fb-a0c27c7e118d
# ╟─75fefe3f-2170-4231-a676-f2b8aef49100
# ╟─eaef6efb-d614-4b54-8599-e5173ac591ab
# ╟─019f57eb-ee5f-4500-958d-830186d62814
# ╟─80a04a89-b23e-4d2f-9a4d-ce887ba0cf61
# ╠═bf17466e-b5a3-4180-acfb-362c5da55d70
# ╠═1a5e09f6-0b83-4e13-af89-4364e219ff16
# ╠═fe609884-c598-4391-ab06-4ba7166b65ff
# ╠═f6119a25-00e6-4a65-a3d8-8a56e8719fb5
# ╠═f1233e72-adc0-43dc-8796-055cf9baf4e7
# ╠═185cdee9-84e7-443f-a6a3-7ae6369e956e
# ╟─ff118324-f687-44d6-aca9-dc84802767e2
# ╟─73b4d226-4062-44d7-9f23-b16864b93b31
# ╟─8c727358-0695-43ae-96b2-ded282e0dedf
# ╟─370b5a64-9fcb-4431-99c1-d0dee3f64932
# ╟─21c26eca-5896-4a67-801a-86a228edc2c5
# ╟─a20538f8-1edb-44dc-bee9-989dc6926ee9
# ╟─7d5c48ff-5806-4bb2-acdd-98a508627bac
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
