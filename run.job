#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:2
#SBATCH --job-name=TEST
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=64000M
#SBATCH --output=TEST_%A.out

module purge
module load 2021
module load Anaconda3/2021.05


source activate rs


# Training
# python3 ./DiffRec/main.py --data_path ./datasets/yelp_clean/ # DONE
# python3 ./L-DiffRec/main.py --data_path ./datasets/yelp_clean/ --sampling_steps 0 --steps 100 # TODO not yet working sampling steps issue
# python3 ./LT-DiffRec/main.py --data_path ./datasets/yelp_clean/ --sampling_steps 0 --steps 100 # TODO not yet working sampling steps issue


# Inference only
# python ./DiffRec/inference.py --data_path ./datasets/ --dataset=yelp_noisy --cuda --gpu=1 # DONE
python ./T-DiffRec/inference.py --data_path ./datasets/ --dataset=yelp_clean --cuda --gpu=1 # DONE # Cant be run on yelp noisy!
python ./L-DiffRec/inference.py --data_path ./datasets/ --dataset=yelp_noisy --cuda --gpu=1 --sampling_steps 0 --steps 100 # DONE, but paras modified; Cant be run on yelp noisy!
python ./LT-DiffRec/inference.py --data_path ./datasets/ --dataset=yelp_clean --cuda --gpu=1 --sampling_steps 0 --steps 100 # DONE, but paras modified


conda deactivate
