# IRM_CDM

## Reproduce results
To reproduce results reported in the paper

please use ./colored_mnist/main_cmnist_plus.py and ./punctuated_sst2/main_psst2_plus.py with hyperparameters reported in the paper.

### PSST-2+

```
# ERM
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 0.0 --color_prob $I --method ERM --cdm_penalty_weight 0.0
done

# IRMBAL
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMBAL --cdm_penalty_weight 0.0
done

# IRMMMD
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    for J in 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
    do
        python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMMMD --mmd_kernel MKMMD --cdm_penalty_weight $J
    done
done

# MMD
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    for J in 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
    do
        python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMMMD --mmd_kernel MKMMD --cdm_penalty_weight $J
    done
done

# IRMCGAN

for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    for J in 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
    do
        python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 7500.0 --color_prob $I --method IRMCGAN --cgan_type KL --cdm_penalty_weight $J
    done
done

# CGAN
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    for J in 0.1 1.0 10.0 100.0 1000.0 10000.0 100000.0
    do
        python ./punctuated_sst2/main_psst2_plus.py -v --lr 0.001 --l2_regularizer_weight 0.0001 --penalty_weight 0.0 --color_prob $I --method CGAN --cgan_type KL --cdm_penalty_weight $J
    done
done
```



### CMNIST+

```
# Default ERM
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./colored_mnist/main_cmnist_plus.py --penalty_anneal_iters 0 --penalty_weight 0.0 --method ERM --cdm_penalty_weight 0.0 --color_prob $I
done

# Default IRM
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./colored_mnist/main_cmnist_plus.py --method IRM --cdm_penalty_weight 0.0 --color_prob $I --penalty_weight 91257.18613115903
done

# Default IRMBAL
for I in 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./colored_mnist/main_cmnist_plus.py --method IRMBAL --cdm_penalty_weight 0.0 --color_prob $I --penalty_weight 91257.18613115903
done

# Oracle
for I in 0.55 0.6 0.65 0.7 0.8 0.85 0.9
do
    python ./colored_mnist/main_cmnist_plus.py --penalty_anneal_iters 0 --penalty_weight 0.0 --method Oracle --cdm_penalty_weight 0.0 --color_prob $I --datadir './colored_mnist/data_local_backup/'
done
```

## Datasets

datasets can be found in ./colored_mnist/data_local and ./punctuated_sst2/data_local
