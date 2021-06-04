"""
Util Functions for Causal Domain Generalization Methods
"""

from typing import Optional, Sequence

import torch
from torch import nn

softmax = nn.Softmax(dim=1)


def calculate_mmd(tr_envs, kernel = "IMQMMD", loss_func = None, downsample=0.05):

    assert(len(tr_envs)==2)

    tr_env0, tr_env1 = tr_envs
    tr_env0, tr_env1 = tr_env0[1], tr_env1[1]
    env0_y1_idx = tr_env0['labels'] == 1
    env0_y0_idx = tr_env0['labels'] == 0
    env1_y1_idx = tr_env1['labels'] == 1
    env1_y0_idx = tr_env1['labels'] == 0

    # len of selected samples
    len_y1 = min(env0_y1_idx.sum(), env1_y1_idx.sum())
    len_y0 = min(env0_y0_idx.sum(), env1_y0_idx.sum())

    if downsample < 1.0:
        len_y1 = int(len_y1*downsample)
        len_y0 = int(len_y0*downsample)

    rand_y1_idx = torch.randperm(len_y1)[:len_y1]
    rand_y0_idx = torch.randperm(len_y0)[:len_y0]

    assert(len_y0>0 and len_y1>0)

    env0_y0_reps = tr_env0['reps'][env0_y0_idx][rand_y0_idx].view(len_y0,-1)
    env0_y1_reps = tr_env0['reps'][env0_y1_idx][rand_y1_idx].view(len_y1,-1)
    env1_y0_reps = tr_env1['reps'][env1_y0_idx][rand_y0_idx].view(len_y0,-1)
    env1_y1_reps = tr_env1['reps'][env1_y1_idx][rand_y1_idx].view(len_y1,-1)

    if kernel == "IMQMMD":
        return calculate_imq_mmd(env0_y0_reps, env0_y1_reps, 
        env1_y0_reps, env1_y1_reps)

    elif kernel == "MKMMD":
        return calculate_mk_mmd(env0_y0_reps, env0_y1_reps, 
        env1_y0_reps, env1_y1_reps, loss_func)

############################### IMQ ############################################

def calculate_imq_mmd(env0_y0_reps, env0_y1_reps, env1_y0_reps, env1_y1_reps):

    h_dim = env0_y0_reps.shape[1]

    # only calculating IMQ_MMD for now
    mmd_loss_y1 = imq_kernel(env0_y1_reps,
                        env1_y1_reps, h_dim=h_dim)
    mmd_loss_y1 = mmd_loss_y1.mean()

    mmd_loss_y0 = imq_kernel(env0_y0_reps,
                        env1_y0_reps, h_dim=h_dim)
    mmd_loss_y0 = mmd_loss_y0.mean()

    mmd_imq_penalty = (mmd_loss_y0 + mmd_loss_y1) / 2.

    return mmd_imq_penalty

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    p2_norm_x = X.pow(2).sum(1).unsqueeze(0)
    norms_x = X.sum(1).unsqueeze(0)
    prods_x = torch.mm(norms_x, norms_x.t())
    dists_x = p2_norm_x + p2_norm_x.t() - 2 * prods_x

    p2_norm_y = Y.pow(2).sum(1).unsqueeze(0)
    norms_y = X.sum(1).unsqueeze(0)
    prods_y = torch.mm(norms_y, norms_y.t())
    dists_y = p2_norm_y + p2_norm_y.t() - 2 * prods_y

    dot_prd = torch.mm(norms_x, norms_y.t())
    dists_c = p2_norm_x + p2_norm_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats

############################# MK #################################

def calculate_mk_mmd(env0_y0_reps, env0_y1_reps, env1_y0_reps, env1_y1_reps, loss_func):

    mmd_loss_y1 = loss_func(env0_y1_reps,
                        env1_y1_reps)
    # mmd_loss_y1 = mmd_loss_y1.mean()

    mmd_loss_y0 = loss_func(env0_y0_reps,
                        env1_y0_reps)
    # mmd_loss_y0 = mmd_loss_y0.mean()

    mmd_mk_penalty = (mmd_loss_y0 + mmd_loss_y1) / 2.

    return mmd_mk_penalty

############################### CGAN LOSS ############################################
def calculate_cgan_loss(tr_envs, D, weights, training=True, cgan_type="KL"):
    """
    gan_type: the type of cgan loss, select from (JS, KL, CDAN)
    """

    if cgan_type == "KL":
        return calculate_kl_cgan_loss(tr_envs, D)
    elif cgan_type == "JS":
        return calculate_js_cgan_loss(tr_envs, D, weights)
    # elif gan_type == "CDAN":
    #     return calculate_cdan_loss(tr_envs)
    else:
        raise NotImplementedError("Choose a valid cgan_type for the GAN loss.")

def calculate_kl_cgan_loss(tr_envs, D):
    kl_cgan_loss = 0.0
    for env_idx, env in enumerate(tr_envs):
        env = env[1]
        target_onehot = nn.functional.one_hot(env['labels'].long()).float()

        concat_reps = torch.cat([env['reps'], target_onehot], 1)
        env_logits = D(concat_reps) #n x 2

        # [D(F(X),OBJ=0,Y),D(F(X),OBJ=1,Y)]
        output = softmax(env_logits)

        # batch_size = reps.shape[0]
        
        kl_cgan_loss_env = - torch.log(output[:,env_idx])
        env['kl_cgan_loss'] = kl_cgan_loss_env.mean()
        kl_cgan_loss += env['kl_cgan_loss']
        
    return kl_cgan_loss

def calculate_js_cgan_loss(tr_envs, D, weights):
    js_cgan_loss = 0.0

    for env_idx, env in enumerate(tr_envs):
        env = env[1]
        target_onehot = nn.functional.one_hot(env['labels'].long()).float()

        concat_reps = torch.cat([env['reps'], target_onehot], 1)
        env_logits = D(concat_reps) #n x 2

        # [D(F(X),OBJ=0,Y),D(F(X),OBJ=1,Y)]
        output = softmax(env_logits)

        # batch_size = reps.shape[0]
        
        kl_cgan_loss_env = - torch.log(output[:,env_idx])

        js_cgan_loss_env = kl_cgan_loss_env - (
                weights[0,env['labels'].long()] * torch.log(1-output[:,0])
                + weights[1,env['labels'].long()] * torch.log(1-output[:,1]))
        env['js_cgan_loss'] = js_cgan_loss_env.mean()

        js_cgan_loss += env['js_cgan_loss']

    return js_cgan_loss

# def calculate_cdan_loss(self,reps,target,env):
    
#     env0_idx = env == 0
#     env1_idx = env == 1

#     target_env0 = target[env0_idx].view(-1,1)
#     target_env1 = target[env1_idx].view(-1,1)

#     reps_env0 = reps[env0_idx]
#     reps_env1 = reps[env1_idx]

#     assert(self.cgan_loss_func is not None)

#     cdan_loss = - self.cgan_loss_func(target_env0, reps_env0, target_env1, reps_env1)
#     output = self.D(reps) # note that in CDAN the self.D has a sigmoid layer at the end
#     output = torch.cat([1-output,output],1)

#     return cdan_loss, output