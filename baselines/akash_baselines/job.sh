#!/bin/bash
# Job name:
#SBATCH --job-name=scan2cad_apr_29_2020
#
# Account:
#SBATCH --account=fc_vivelab
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akashgokul@berkeley.edu
## Command(s) to run (example):
module load python
#!/bin/bash
# Job name:
#SBATCH --job-name=scan2cad_apr_29_2020
#
# Account:
#SBATCH --account=fc_vivelab
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Request one node:
#SBATCH --nodes=1
#
# Request cores (24, for example)
#SBATCH --ntasks-per-node=2
#
#Request GPUs
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=72:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=akashgokul@berkeley.edu
## Command(s) to run (example):
module load python
module load cuda/10.1
source activate myRAISEenv
python3 SACV2_run.py --extra_train 101 --response_type threshold_exp

