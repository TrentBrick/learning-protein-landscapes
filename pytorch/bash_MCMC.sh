#!/bin/bash
#SBATCH -c 20                               # Request one core
#SBATCH -N 1                               # Request one node (if you request m$
                                           # -N 1 means all cores will be on th$
#SBATCH -t 1-12:59                         # Runtime in D-HH:MM format
#SBATCH -p priority                           # Partition to run in
#SBATCH --mem=20G                          # Memory total in MB (for all cores)
#SBATCH -o hill_slurm/MCMC-slurm-%j.out                 # File to which STDOUT + ST$
hostname
pwd
srun stdbuf -oL -eL ~/anaconda3/bin/python run.py --exp_base_name MCMC_fullProtein_12Ksteps \
--run_model main_MCMC.py --protein_len 0 --nwalkers 64 --nsteps 600 --ncores 20 --print_every 200