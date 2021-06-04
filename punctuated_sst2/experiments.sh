#!/usr/bin/env bash

#
# Scripts for reproducing PunctuatedSST-2 experiments in the paper.
#

# Create environments
# python make_environments.py --datadir ./SST --destdir data/PunctuatedSST-2 --version default
# python make_environments.py --datadir ./SST --destdir data/GrayscaleSST-2 --version grayscale

# Default ERM with grayscale data (with hyperparameter search)
# python main.py -v --datadir data/GrayscaleSST-2 --penalty_anneal_iters 0 --penalty_weight 0.0 --lr 0.01 --l2_regularizer_weight 0.0005

# Default ERM (with hyperparameter search)
# python main.py -v --penalty_anneal_iters 0 --penalty_weight 0.0 --lr 0.01 --l2_regularizer_weight 0.01

# Default IRM (with hyperparameter search)
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I
# done

# ERM
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 0.0 --color_prob $I --method ERM --cdm_penalty_weight 0.0
done

# IRMBAL
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMBAL --cdm_penalty_weight 0.0
# done

# IRMMMD
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     for J in 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
#     do
#         python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMMMD --mmd_kernel MKMMD --cdm_penalty_weight $J
#     done
# done

# MMD
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     for J in 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
#     do
#         python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMMMD --mmd_kernel MKMMD --cdm_penalty_weight $J
#     done
# done

# # IRMCGAN

# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     for J in 100000.0
#     do
#         python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMCGAN --cgan_type KL --cdm_penalty_weight $J
#     done
# done

# # CGAN
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     for J in 100000.0
#     do
#         python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 0.0 --color_prob $I --method CGAN --cgan_type KL --cdm_penalty_weight $J
#     done
# done


