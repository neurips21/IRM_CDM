#!/usr/bin/env bash

#
# Scripts for reproducing the ColoredMNIST experiments in the paper.
#

################################################################################

# Default ERM
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     python ./colored_mnist/main_cmnist_plus.py --penalty_anneal_iters 0 --penalty_weight 0.0 --method ERM --cdm_penalty_weight 0.0 --color_prob $I
# done

# Default IRM
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     python ./colored_mnist/main_cmnist_plus.py --method IRM --cdm_penalty_weight 0.0 --color_prob $I --penalty_weight 91257.18613115903
# done

# Default IRMBAL
# for I in 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     python ./colored_mnist/main_cmnist_plus.py --method IRMBAL --cdm_penalty_weight 0.0 --color_prob $I --penalty_weight 91257.18613115903
# done

# Oracle
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
for I in 0.9
do
    python ./colored_mnist/main_cmnist_plus.py --penalty_anneal_iters 0 --penalty_weight 0.0 --method Oracle --cdm_penalty_weight 0.0 --color_prob $I --datadir './colored_mnist/data_local_backup/'
done


################################################################################
# IRMCGAN
# for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
# do
#     for J in 0.01 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
#     do
#         python ./colored_mnist/main_cmnist_plus.py -v --color_prob $I --method IRMCGAN --cgan_type KL --cdm_penalty_weight $J
#     done
# done