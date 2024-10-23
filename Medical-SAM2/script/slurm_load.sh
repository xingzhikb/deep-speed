#!/bin/bash

module load git
module load cuda
module load nccl
module load gcc
module load cmake
module load conda

conda activate medsam2

cd /blue/weishao/ywan1332.ucsc/zc/code/medical-sca/Medical-SAM2

sbatch ./script/train_3d_sca_s.sh