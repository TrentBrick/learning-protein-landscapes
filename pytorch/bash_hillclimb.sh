#!/bin/bash
#SBATCH -c 20                               # Request one core
#SBATCH -N 1                               # Request one node (if you request m$
                                           # -N 1 means all cores will be on th$
#SBATCH -t 0-11:59                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=20G                          # Memory total in MB (for all cores)
#SBATCH -o hill_slurm/hill-slurm-%j.out                 # File to which STDOUT + ST$
hostname
pwd
srun stdbuf -oL -eL ~/anaconda3/bin/python run.py --exp_base_name test_tenSteps \
--run_model main_climb.py --protein_len 0 --nwalkers 64 --nsteps 10 --ncores 20 --print_every 200