#!/bin/bash 
## Submit job to our class's allocation
##SBATCH --partition=sla-prio
#SBATCH --partition=burst
#SBATCH --qos=burst4x 
#SBATCH --account=ebf11-fa23_p_gpu

## Time requested: 0 hours, 15 minutes, 0 seconds
#SBATCH --time=0:15:00 

## Ask for one core of one node, plus 1 CPU
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1 
#SBATCH --gpus=1

## Promise that each processor will use no more than 1GB of RAM
#SBATCH --mem-per-cpu=8GB

## Save STDOUT and STDERR into one file (%j will expand to become the SLURM job id)
#SBATCH --output=ex1_%j.log
## Optionally could uncomment line below to write STDERR to a separate file
##SBATCH --error=ex1_%j.stderr  

## Specificy job name, so easy to find using squeue –u
#SBATCH --job-name=ex1

## Uncomment next two lines (by removing one of #'s in each line) and replace with your email if you want to be notifed when jobs start and stop
##SBATCH --mail-user=YOUR_EMAIL_HERE@psu.edu
## Ask for emails when jobs begins, ends or fails (options are ALL, NONE, BEGIN, END, FAIL)
#SBATCH --mail-type=ALL

echo "Starting job $SLURM_JOB_NAME"
echo "Job id: $SLURM_JOB_ID"
date

echo "This job was assigned the following nodes"
echo $SLURM_NODELIST

echo "Activing environment with that provides Julia 1.9.2"
source /storage/group/RISE/classroom/astro_528/scripts/env_setup

echo "About to change into $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR            # Change into directory where job was submitted from

# env |grep SLURM               # Uncomment if you want to see all the environment variables set

FILE=./Project_has_been_instantiated
if [ -f "$FILE" ]; then
    echo "# $FILE exists.  Assuming no need to instantiate project to install packages."
else 
    echo "# $FILE does not exist. Will install relevant packages."
    julia --project -e 'import Pkg; Pkg.instantiate() '
    julia --project -e 'import Pluto, Pkg; Pluto.activate_notebook_environment("ex1.jl"); Pkg.instantiate() '
    touch $FILE
fi

date
echo "# About to run Pluto notebook and generate HTML version with outputs, using $SLURM_TASKS_PER_NODE CPU cores on one node"
# export JULIA_PKG_OFFLINE="true"
julia --project -t $SLURM_TASKS_PER_NODE -e 'import Pkg, PlutoSliderServer; Pkg.offline(); PlutoSliderServer.export_notebook("ex1.jl")'
# To just run without saving HTML version
#julia -t $SLURM_TASKS_PER_NODE -e 'using Pkg, Pluto; Pkg.offline(); Pluto.activate_notebook_environment("ex1.jl"); Pkg.instantiate(); include("ex1.jl")'  
echo "Julia exited"
date
