"""
Minimal IRM for ColoredSST-2 with a bag-of-words model
"""

import argparse

import csv
import os
import tempfile

from http.client import METHOD_NOT_ALLOWED
import itertools as it
from typing import List

import numpy as np
from numpy import add, random
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import torchtext

from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy

from models import BOWClassifier, Discriminator
from data_processors import get_train_examples, get_test_examples
from causal_utils import calculate_mmd, calculate_cgan_loss


def mean_nll(logits, y):
    return nn.functional.cross_entropy(logits, y)


def mean_accuracy(logits, y):
    preds = torch.argmax(logits, dim=1).float()
    return (preds == y).float().mean()


def penalty(logits, y):
    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def convert_examples_to_features(
        examples: List[dict],
        vocab: torchtext.vocab.Vectors,
        device: torch.device,
        method: str,
        training: bool = True,
):
    """Convert examples to torch.Tensors of (text, offsets, labels)."""
    
    if method == "IRMBAL" and training:
        n_y1 = 0
        n_y0 = 0
        for example in examples:
            if int(example['label']) == 1:
                n_y1 += 1
            else:
                n_y0 += 1
        
        n_downsample = min(n_y1,n_y0)
        prob_y1 = n_downsample / float(n_y1)
        prob_y0 = n_downsample / float(n_y0)

        rand = np.random.rand(len(examples))

        examples_ = []
        for idx, example in enumerate(examples):
            if int(example['label']) == 1 and rand[idx] < prob_y1:
                examples_.append(example)
            if int(example['label']) == 0 and rand[idx] < prob_y0:
                examples_.append(example)
    else:
        examples_ = examples
    
    text, offsets, labels = [], [], []
    current_offset = 0
    for example in examples_:
        # input
        words = example['text'].split()
        word_ids = [vocab.stoi[word] for word in words if word in vocab.stoi]
        if len(word_ids) < 1:
            continue
        text.extend(word_ids)
        offsets.append(current_offset)
        current_offset += len(word_ids)
        # label
        labels.append(int(example['label']))
    return {
        'text': torch.tensor(text).to(device),
        'offsets': torch.tensor(offsets).to(device),
        'labels': torch.tensor(labels).to(device),
    }


def run_punctuated_sst2(
    azure: bool,
        datadir: str,
        resdir: str,
        method: str,
        mmd_kernel: str,
        cgan_type: str,
        color_prob: float,
        imba: float,
        glove_name: str = "6B",  # 6B, 42B, 840B, twitter.27B
        glove_dim: int = 300,
        n_layers: int = 3,
        l2_regularizer_weight: float = 0.001,
        lr: float = 0.001,
        n_restarts: int = 5,
        penalty_anneal_iters: int = 100,
        penalty_weight: float = 10000.0,
        cdm_penalty_weight: float = 0.0,
        mmd_sample: float = 0.01, #sample the dataset to fit into the GPU RAM for MMD
        steps: int = 501,
        track_best: bool = False,
        verbose: bool = False
):
    """Run PunctuatedSST-2 experiment and return train/test accuracies."""

    if method in ["ERM", "MMD", "CGAN", "REx"]:
        assert(penalty_weight==0)
    if method in ["IRM", "IRMBAL", "IRMMMD", "IRMCGAN"]:
        assert(penalty_weight>0)
        assert(penalty_anneal_iters>=0)
    if method in ["MMD", "CGAN", "IRMMMD", "IRMCGAN"]:
        assert(cdm_penalty_weight>0)
    if method in ["ERM", "IRM", "IRMBAL", "REx"]:
        assert(cdm_penalty_weight==0)
    
    if mmd_kernel == "MKMMD":
        # using the default setting for this
        mmd_loss_func = MultipleKernelMaximumMeanDiscrepancy(
        kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
        linear=False, quadratic_program=False)
    else:
        mmd_loss_func = None
    
    assert(cgan_type in {"KL","JS","CDAN"})

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab (a torchtext.vocab.Vectors object)

    suffix = 'PSST2_PLUS'+'_color_prob_' + str(color_prob) + '_imba_' + str(imba)

    if azure:
        datadir_ = datadir
    else:
        # save the results for the same dataset in one file
        datadir_ = datadir + suffix
    
    print("let's see how the directory looks like.")
    print(os.listdir('./'))
    print(datadir_)
    print(os.listdir(datadir_))

    train_examples = get_train_examples(datadir_)
    test_examples = get_test_examples(datadir_)

    vocab = torchtext.vocab.GloVe(name=glove_name, dim=glove_dim)
    embeddings = vocab.vectors

    # Prepare environments
    n_classes = 2
    train_envs = {env_name: convert_examples_to_features(examples, vocab, device, method, training=True)
                  for env_name, examples in train_examples.items()}
    test_envs = {env_name: convert_examples_to_features(examples, vocab, device, method, training=False)
                 for env_name, examples in test_examples.items()}
    all_envs = [env_name for env_name in it.chain(train_envs, test_envs)]

    final_accs = {env_name: [] for env_name in all_envs}
    best = [{'step': 0, 'min_acc': 0.0, 'loss': 0.0}
            for _ in range(n_restarts)]

    if not os.path.isdir(resdir):
        os.mkdir(resdir)

    if not os.path.isdir(os.path.join(resdir, suffix)):
        os.mkdir(os.path.join(resdir, suffix))

    file_name = method + 'lr' + str(lr)
    if method in ["IRM", "IRMMMD", "IRMCGAN", "IRMBAL"]:
        file_name += 'pen'+str(penalty_weight)+'iter'+str(penalty_anneal_iters)
    if method in ["IRMMMD", "IRMCGAN", "MMD", "CGAN"]:
        file_name += 'cdm_pen'+str(cdm_penalty_weight)
    if method in ["MMD", "IRMMMD"]:
        file_name += 'mmd_kernel'+mmd_kernel
    if method in ["CGAN", "IRMCGAN"]:
        file_name += 'cgan_type'+cgan_type
    file_name += '.csv'

    of = open(os.path.join(resdir, suffix, file_name),'w')

    wrt = csv.writer(of)

    for restart in range(n_restarts):

        # Initialize model
        model = BOWClassifier(embeddings, n_layers, n_classes).to(device)

        if method in ["CGAN", "IRMCGAN"]:
            D = Discriminator(glove_dim).to(device)
        else:
            D = None

        if verbose and restart == 0:
            print(model)
            print("# trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # Train loop
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if verbose:
            pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test0 acc', 'test1 acc', 'test_ood acc')

        best_min_acc, best_loss, best_step = 0.0, 0.0, 0

        for step in range(steps):
            for _, env in it.chain(train_envs.items(), test_envs.items()):

                X_env = env['text']
                y_env = env['labels']
                offsets_env = env['offsets']

                reps, logits = model(X_env, offsets_env)  # multi-class logit
                env['nll'] = mean_nll(logits, y_env)
                env['acc'] = mean_accuracy(logits, y_env)
                env['penalty'] = penalty(logits, y_env)
                env['reps'] = reps

            train_nll = torch.stack([env['nll'] for _, env in train_envs.items()]).mean()
            train_acc = torch.stack([env['acc'] for _, env in train_envs.items()]).mean()
            train_penalty = torch.stack([env['penalty'] for _, env in train_envs.items()]).mean()

            # L2 norm
            weight_norm = torch.tensor(0.).cuda()
            for w in model.parameters():
                if w.requires_grad:
                    weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += l2_regularizer_weight * weight_norm
            annealed_penalty_weight = (penalty_weight
                if step >= penalty_anneal_iters else 1.0)
            
            if method in ["IRM", "IRMCGAN", "IRMMMD"]:
                loss += annealed_penalty_weight * train_penalty

                if annealed_penalty_weight > 1.0:
                    # Rescale the entire loss to keep gradients in a reasonable range
                    loss /= annealed_penalty_weight
            
            if method in ["MMD", "IRMMMD"]:
                cdm_penalty = calculate_mmd(train_envs.items(), kernel=mmd_kernel, loss_func=mmd_loss_func, downsample=mmd_sample)

            elif method in ["CGAN", "IRMCGAN"]:
                cdm_penalty = calculate_cgan_loss(train_envs.items(), D, cgan_type=cgan_type)

            else:
                cdm_penalty = 0
            
            loss += cdm_penalty * cdm_penalty_weight        
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # monitor stats at min_acc for hyperopt (best or last)
            min_acc = min(env['acc'].item() for _, env in test_envs.items())
            if not track_best or min_acc > best_min_acc:
                best_min_acc = min_acc
                best_loss = loss.item()
                best_step = step
                best[restart].update({
                    'step': step,
                    'min_acc': best_min_acc,  # minimum of test accuracies
                    'loss': best_loss,  # training loss
                    'train0_acc': train_envs['train0']['acc'].item(),
                    'train1_acc': train_envs['train1']['acc'].item(),
                    # 'test0_acc': test_envs['test0']['acc'].item(),
                    # 'test1_acc': test_envs['test1']['acc'].item(),
                    # 'test_ood_acc': test_envs['test_ood']['acc'].item(),
                })

                for env_idx, test_env in enumerate(test_envs):
                    best[restart].update({
                        test_env+'_acc': test_envs[test_env]['acc'].item()
                    })

            if verbose and step % 100 == 0:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    # test_envs['test0']['acc'].detach().cpu().numpy(),
                    # test_envs['test1']['acc'].detach().cpu().numpy(),
                    # test_envs['test_ood']['acc'].detach().cpu().numpy(),
                )

                for env_idx, test_env in enumerate(test_envs): 
                    pretty_print(
                        test_envs[test_env]['acc'].detach().cpu().numpy()
                    )

        for env_name in train_envs:
            final_accs[env_name].append(train_envs[env_name]['acc'].item())
        for env_name in test_envs:
            final_accs[env_name].append(test_envs[env_name]['acc'].item())
        if verbose:
            accs = ", ".join(f"{env_name} {best[restart][f'{env_name}_acc']:.5f}"
                             for env_name in all_envs)
            print(f'Restart {restart}: {accs}, '
                  f"min test acc {best[restart]['min_acc']:.5f} (step {best_step})")

    print(f'[Accuracies at best minimum test set accuracy over {n_restarts} restarts]')
    pretty_print("env_name", "mean", "std")

    for env_name in all_envs:
        best_accs = [best[restart][f'{env_name}_acc'] for restart in range(n_restarts)]
        mean, std = np.mean(best_accs), np.std(best_accs)
        pretty_print(env_name, mean, std)
        wrt.writerow([env_name, mean, std])
        
    best_or_last = "Best" if track_best else "Final"
    print(f'[{best_or_last} minimum test set accuracy over {n_restarts} restarts]')
    best_min_accs = [best[restart]['min_acc'] for restart in range(n_restarts)]
    mean, std = np.mean(best_min_accs), np.std(best_min_accs)
    pretty_print("mean", "std")
    pretty_print(mean, std)

    of.close()

    return best


def main():
    parser = argparse.ArgumentParser(description='Minimal PunctuatedSST-2')
    
    parser.add_argument('--azure', type=bool, default=False)
    parser.add_argument('--datadir', type=str, default='./punctuated_sst2/data_local/',
                        help='directory containing PunctuatedSST-2 datasets')
    
    parser.add_argument('--resdir', type=str, default='./punctuated_sst2/results/',
                        help='directory containing PunctuatedSST-2 datasets')
    
    parser.add_argument('--color_prob', type=float, default=0.8, help='P(C | Y,E) of the tr envs')
    parser.add_argument('--imba', type=float, default=0.9, help='class imba of tr envs')
    parser.add_argument('--method', type=str, default='IRMMMD')
    parser.add_argument('--mmd_kernel', type=str, default="MKMMD")
    parser.add_argument('--cgan_type', type=str, default='KL')

    parser.add_argument('--glove_name', type=str, default='6B',
                        help='name specifying GloVe vectors (default: 6B)')
    parser.add_argument('--glove_dim', type=int, default=300,
                        help='')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=7500.0)
    parser.add_argument('--cdm_penalty_weight', type=float, default=0.1)

    parser.add_argument('--mmd_sample', type=float, default=0.01)

    parser.add_argument('--steps', type=int, default=501)
    parser.add_argument('--track_best', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    print(args)
    
    run_punctuated_sst2(**vars(args))

if __name__ == '__main__':
    main()
