#!/bin/bash
# module load anaconda/2021.11 compilers/cuda/12.1 compilers/gcc/11.3.0 cudnn/8.8.1.3_cuda12.x
# source activate flatness
module load  compilers/cuda/11.8 compilers/gcc/9.3.0 cudnn/8.4.0.27_cuda11.x anaconda
source activate flatness310
source offline_mode.sh

python scripts/flan_t5_layer_wise_adamerging.py \
    models=flan-t5-large peft=lora-16 fast_dev_run=false batch_size=4
