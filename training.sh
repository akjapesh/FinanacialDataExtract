#!/bin/bash

#SBATCH -J CasRel            #Job name(--job-name)
#SBATCH -o logs/training_NYT.log          #Name of stdout output file(--output)
#SBATCH -e logs/training_NYT.log   #Name of stderr error file(--error)
#SBATCH -p gpu              #Queue (--partition) name
#SBATCH -n 1                    #Total Number of mpi tasks (--ntasks .should be 1 for serial)
#SBATCH --gres=gpu:2
#SBATCH -c 10                    #(--cpus-per-task) Number of Threads
#SBATCH --mem=32000
#SBATCH --mail-user=arusarkabose@gmail.com        # user's email ID where job status info will be sent
#SBATCH --mail-type=ALL        # Send Mail for all type of event regarding the job
#SBATCH --time 3-0            # 3 days max

module load compiler/intel-mpi/mpi-2019-v5
module load compiler/cudnn/7.6.2
module load compiler/cuda/10.1

#source /home/$USER/.bashrc
# source /home/$USER/.bash_aliases

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python --version
nvcc --version
python run.py --train=True --dataset=FinRel
python run.py --dataset=FinRel
