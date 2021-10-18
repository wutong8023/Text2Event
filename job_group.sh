#!/bin/bash

#SBATCH -t 100800
#SBATCH -N 1
#SBATCH --gres=gpu:1

# Memory usage (MB)
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128000

#SBATCH --mail-user=wutong8023@163.com
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# IMPORTANT!!! check the job name!
#SBATCH -o log/%J-fsl_fine_shot-15_few_shot_15_2021-10-18-12-41.out
#SBATCH -e log/%J-fsl_fine_shot-15_few_shot_15_2021-10-18-12-41.err
#
#
#
#
#SBATCH -J fsl_fine_shot-15_few_shot_15_2021-10-18-12-41
#
#
#
module load python3
source /home/tongwu/envs/prefixEE/bin/activate
module load cuda-11.2.0-gcc-10.2.0-gsjevs3



bash run_seq2seq_verbose_prefix.bash -d 0 -f tree -m t5-base --label_smoothing 0 -l 5e-5 --lr_scheduler linear --warmup_steps 2000 -b 16 --epoch 60 --data oneie/few_shot_15 --output_dir models/fsl_fine_shot-15_few_shot_15_2021-10-18-12-41 --tuning_type fine   --pltf gp

