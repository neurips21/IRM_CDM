#!/usr/bin/python3 -u

"""
Create environments for CMNIST+.

"""

import argparse
from collections import Counter
import numpy as np
import os.path
import string
from torchvision import datasets
from torchvision.utils import save_image
import torch

import cv2


from utils import read_raw_data, write_processed_image_data

def save_image_local(images_env,bin_labels_env,env_idx):
    images_env = images_env
    perm = torch.randperm(images_env.shape[0])
    idx = perm[:64]
    tensor_images_env = torch.FloatTensor(images_env[idx])
    save_image(tensor_images_env,
                            './images_env'+str(env_idx)+'.png', nrow=8)
    print("targets env {}".format(env_idx))
    print(bin_labels_env[:64].reshape(8,-1))

def color_image(X, artifact_type, X_c):
    """Color (perturb) sentences according to spurious artifacts.

    Artifact label is assumed to be binary.

    [artifact types]
    default: spurious punctuation (contains period or exclamation mark)
    grayscale: return original sentence (no artifact)
    """
    # assert isinstance(X, tuple)
    assert artifact_type in {"default", "grayscale"}
    assert X_c[0] in {0, 1, 2}

    if artifact_type == "grayscale":
        return X

    X_ = np.stack([X,X,X],axis=1)

    for i in range(3):
        idxes_put_to_zero = np.arange(len(X))[X_c != i]
        X_[idxes_put_to_zero, i, :, :] *= 0

    return X_

def split_envs_tr(X,y,imba_ratio=0.9):
    # input is 60k+ training images
    # return data with P(Y=1|E=1) = args.imba and P(Y=1|E=2) = 1-args.imba
    # The causal relationship is Y -> E

    X_c1 = []
    X_c0 = []

    for i in range(len(y)):
        if y[i] == 1:
            X_c1.append(X[i])
        else:
            X_c0.append(X[i])

    y = np.array(y)

    y_c1 = y[y == 1]
    y_c0 = y[y == 0]

    major_size_c1 = int(imba_ratio*len(X_c1))
    minor_size_c1 = len(X_c1) - major_size_c1
    major_size_c0 = int(imba_ratio*len(X_c0))
    minor_size_c0 = len(X_c0) - major_size_c0

    minor_idx_c1 = np.random.choice(len(X_c1),minor_size_c1,replace=False)
    major_idx_c1 = np.array([i for i in range(len(X_c1)) if i not in minor_idx_c1])
    minor_idx_c0 = np.random.choice(len(X_c0),minor_size_c0,replace=False)
    major_idx_c0 = np.array([i for i in range(len(X_c0)) if i not in minor_idx_c0])

    minor_idx_c1_set = set(minor_idx_c1.tolist())
    major_idx_c1_set = set(major_idx_c1.tolist())
    minor_idx_c0_set = set(minor_idx_c0.tolist())
    major_idx_c0_set = set(major_idx_c0.tolist())

    print("Size of Major Idx C1 {}, Minor Idx C1 {}".format(len(major_idx_c1),len(minor_idx_c1)))
    print("Size of Major Idx C0 {}, Minor Idx C0 {}".format(len(major_idx_c0),len(minor_idx_c0)))
    print("Just Make Sure They are Roughly the Same.")

    # X_e1 = np.concatenate([X_c1[major_idx_c1],X_c0[minor_idx_c0]],axis=0)
    # X_e2 = np.concatenate([X_c1[minor_idx_c1],X_c0[major_idx_c0]],axis=0)

    X_e1 = []
    X_e2 = []
    # add major c1 to X_e1 and minor c1 to X_e2
    for idx, sent in enumerate(X_c1):
        if idx in major_idx_c1_set:
            X_e1.append(sent)
        elif idx in minor_idx_c1_set:
            X_e2.append(sent)
    
    for idx, sent in enumerate(X_c0):
        if idx in major_idx_c0_set:
            X_e2.append(sent)
        elif idx in minor_idx_c0_set:
            X_e1.append(sent)



    # targets_e1 = np.concatenate([np.ones_like(major_idx_c1),np.zeros_like(minor_idx_c0)])
    # targets_e2 = np.concatenate([np.ones_like(minor_idx_c1),np.zeros_like(major_idx_c0)])

    y_e1 = np.concatenate([y_c1[major_idx_c1], y_c0[minor_idx_c0]])
    y_e2 = np.concatenate([y_c1[minor_idx_c1], y_c0[major_idx_c0]])

    # random permute the images and targets
    random_idx_e1 = np.random.permutation(len(major_idx_c1) + len(minor_idx_c0))
    random_idx_e2 = np.random.permutation(len(major_idx_c0) + len(minor_idx_c1))

    X_e1_rand = []
    X_e2_rand = []
    y_e1_rand = []
    y_e2_rand = []

    for idx in random_idx_e1:
        X_e1_rand.append(X_e1[idx])
        y_e1_rand.append(y_e1[idx])

    for idx in random_idx_e2:
        X_e2_rand.append(X_e2[idx])
        y_e2_rand.append(y_e2[idx])

    # out: { "nameN": {"inputs": inputsN, "labels": labelsN} }
    return {"train0": {"inputs": X_e1, "labels": y_e1}, 
    "train1": {"inputs": X_e2, "labels": y_e2}}

def get_color_probs(args):
    main_color_prob = args.color_prob
    minor_color_prob = (1-args.color_prob)/2

    rgb_probs_env1 = [[minor_color_prob, main_color_prob, minor_color_prob], # P(C|Y=1,E=0)
                        [main_color_prob, minor_color_prob, minor_color_prob]] # P(C|Y=0,E=0)
    rgb_probs_env2 = [[minor_color_prob, minor_color_prob, main_color_prob],
                        [main_color_prob, minor_color_prob, minor_color_prob]]
    rgb_probs_env3 = [[0.8, 0.1, 0.1], [0.2, 0.4, 0.4]]

    return [rgb_probs_env1, rgb_probs_env2, rgb_probs_env3]


def split_envs_ts(inputs, labels,
               name="", n_envs=1, match_lengths=False, rng=None, repeat=False):
    """Randomly split inputs and labels into different environments.

    Optionally matches the number of samples in each environment. (for train)
    """

    """
    repeat means use the same samples to create multiple (test) domains.
    """
    n = len(inputs)
    if rng is None:
        print("warning: RNG not provided, using unknown random seed")
        rng = np.random.RandomState()

    # randomly split into environments
    out = {
        f"{name}{env}": {"inputs": [], "labels": []}
        for env in range(n_envs)
    }

    if repeat:
        for inp, label in zip(inputs, labels):
            for env in range(n_envs):
                env_name = f"{name}{env}"
                out[env_name]["inputs"].append(inp)
                out[env_name]["labels"].append(label)

    else:
        env_indices = rng.randint(n_envs, size=n)
        for inp, label, env in zip(inputs, labels, env_indices):
            env_name = f"{name}{env}"
            out[env_name]["inputs"].append(inp)
            out[env_name]["labels"].append(label)

    # match lengths between environments
    if match_lengths:
        maxlen = max(len(ds["inputs"]) for env_name, ds in out.items())
        for env_name, ds in out.items():
            n_extra = maxlen - len(ds["inputs"])
            if n_extra >= 1:
                extra_indices = rng.choice(len(ds["inputs"]), size=n_extra)
                ds["inputs"] += [ds["inputs"][i] for i in extra_indices]
                ds["labels"] += [ds["labels"][i] for i in extra_indices]

    # out: { "nameN": {"inputs": inputsN, "labels": labelsN} }
    return out


def color_binary_dataset(X, y, artifact_type,
                         flip_label=0.25, p_env=[[0.0,0.0,0.0],[0.0,0.0,0.0]],
                        rng=None):
    """Give artifical "color" tokens to inputs that correlate with the label.

    *Assumed: label is binary.*

    Analogous to the colored MNIST dataset construction in the IRM paper."""

    if rng is None:
        print("warning: RNG not provided, using unknown random seed")
        rng = np.random.RandomState()

    y = np.array(y)
    y = y.astype(np.float)
    print("Flipping Label with Probability {}".format(flip_label))
    y_noisy = np.logical_xor(y,
                        np.random.binomial(1, flip_label, len(y)))

    Xc = np.zeros(len(y)) # Nx1, value \in [0,1,2], representing three types of artifacts
    c1_idx = y_noisy == 1
    c0_idx = y_noisy == 0

    Xc[c1_idx] = np.random.choice([0,1,2],np.sum(c1_idx),
                                      replace=True,p=p_env[0])

    Xc[c0_idx] = np.random.choice([0,1,2],np.sum(c0_idx),
                                      replace=True,p=p_env[1])

    X_colored = color_image(X, artifact_type, Xc)

    return X_colored, y_noisy.astype(np.int)


def color_mnist_plus(args):
    """Generate a CMNIST+ dataset.

    datadir is the location that contains the original CMNIST data."""

    datadir, destdir, version = args.datadir, args.destdir+'_color_prob_'+str(args.color_prob)+'_imba_'+str(args.imba), args.version

    # default setup
    n_envs_tr = 2
    n_envs_ts = 1
    
    label_map = {"0": 0, "1": 1}
    rng = np.random.RandomState(1)

    p_envs = get_color_probs(args)
    p_envs_tr = p_envs[:2]
    p_envs_ts = [p_envs[2]]

    if version == "grayscale":
        artifact_type, flip_label, p_envs = ("grayscale", 0.25, [[0.0, 0.0, 0.0],[0.0,0.0,0.0]])
    else:
        artifact_type, flip_label = ("default", 0.25)

    # train: train0(p=0.2), train1(p=0.1)
    # X_tr, y_tr = read_raw_data(
    #     os.path.join(datadir, "train.tsv"), ["sentence"], "label", label_map
    # )

    mnist = datasets.MNIST(datadir, train=True, download=True)
    X_tr, y_tr_raw = mnist.data[:50000].numpy(), mnist.targets[:50000].numpy()
    X_ts, y_ts_raw = mnist.data[50000:].numpy(), mnist.targets[50000:].numpy()

    X_tr = X_tr.reshape((-1, 28, 28))[:, ::2, ::2]
    X_ts = X_ts.reshape((-1, 28, 28))[:, ::2, ::2]

    y_tr = (y_tr_raw > 4).astype(np.int)
    y_ts = (y_ts_raw > 4).astype(np.int)
    
    # get imbalanced classes for the two tr envs
    train = split_envs_tr(X_tr, y_tr, imba_ratio=args.imba)

    for env in range(n_envs_tr):
        ctr = Counter(train[f"train{env}"]["labels"])
        majority_ratio = ctr.most_common(1)[0][1] / sum(ctr.values())
        print(f"train{env}:", ctr, ", majority:", majority_ratio)
    
    train0, train1 = [
        color_binary_dataset(train[f"train{env}"]["inputs"],
                             train[f"train{env}"]["labels"],
                             artifact_type,
                             flip_label=flip_label,
                             p_env=p_env,
                             rng=rng)
        for env, p_env in enumerate(p_envs_tr)
    ]
    
    test = split_envs_ts(
        X_ts, y_ts,
        name="test", n_envs=n_envs_ts, match_lengths=False, rng=rng, repeat=False
    )

    for env in range(n_envs_ts):
        ctr = Counter(test[f"test{env}"]["labels"])
        majority_ratio = ctr.most_common(1)[0][1] / sum(ctr.values())
        print(f"test{env}:", ctr,
              ", majority:", majority_ratio)

    test_envs = [
        color_binary_dataset(test[f"test{env_idx}"]["inputs"],
                             test[f"test{env_idx}"]["labels"],
                             artifact_type,
                             flip_label=flip_label,
                             p_env=p_env,
                             rng=rng)
        for env_idx, p_env in enumerate(p_envs_ts)
    ]

    outputs = {
        "train0": train0,
        "train1": train1
    }

    for env_idx, test_env in enumerate(test_envs):
        outputs[f"test{env_idx}"] = test_env

    save_image_local(outputs['train0'][0],outputs['train0'][1],0)
    save_image_local(outputs['train1'][0],outputs['train1'][1],1)
    save_image_local(outputs['test0'][0],outputs['test0'][1],2)

    write_processed_image_data(outputs, destdir)

    # return train0, train1, test0, test1, test_ood


def main():
    parser = argparse.ArgumentParser(description="make environments for PunctuatedSST-2")
    parser.add_argument('--datadir', default="./colored_mnist/", help="directory containing raw data")
    parser.add_argument('--destdir', default="./colored_mnist/data_local_backup/CMNIST_PLUS", help="output directory")
    parser.add_argument('--version', default="default",
                        help="dataset version (default or grayscale)")

    parser.add_argument('--color_prob', default=0.9,
                        help="rho in the paper")
    parser.add_argument('--imba', default=0.9,
                        help="label imba")
                        
    args = parser.parse_args()
    color_mnist_plus(args)


if __name__ == "__main__":
    main()