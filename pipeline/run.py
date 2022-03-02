'''
'''
from methods.baselinefinetune import BaselineFinetune
import FSLTask
import random
import data.feature_loader as feat_loader
from io_utils import model_dict, \
                     parse_args, \
                     get_resume_file, \
                     get_best_file, \
                     get_assigned_file
from collections import Counter
from pipeline.free_lunch_util import normalize_l2, test_DC, NC, NCs, NC_o, score_ens
from evaluate_DC import distribution_calibration
from sklearn.linear_model import LogisticRegression
import torch
import os
import pickle
import numpy as np
import time
import sys
import copy
import json
import argparse
from tqdm import trange

def param_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname',
                        default='miniImagenet', help="dataset name")
    parser.add_argument('--method',
                        default='s2m2', help="method should be s2m2|rot|mix")
    parser.add_argument('--n_ways',
                        default=5, type=int, help="n_ways")
    parser.add_argument('--n_shot',
                        default=5, type=int, help="n_shot")
    parser.add_argument('--n_queries',
                        default=15, type=int, help="n_queries")
    parser.add_argument('--n_runs',
                        default=1000, type=int, help="n_runs")
    parser.add_argument('--dgr',
                        default='', help="degree can be 90 180 270")
    parser.add_argument('--save_dist', action='store_true',
                        default=False, help="save the distances to pkl")
    parser.add_argument('--beta',
                        default=0.5, type=float, help="beta")
    parser.add_argument('--norm',
                        default='True', help="L2-norm")
    parser.add_argument('--combine', action='store_true',
                        default=False, help="simply combine all rotations")
    parser.add_argument('--re_size' ,
                        default=84, type=int, help='img size ')
    parser.add_argument('--flip',
                        default='False', help="flipping")
    parser.add_argument('--nv',
                        default='novel', help="novel or val")
    parser.add_argument('--transform',
                        default='beta', help="beta or log")
    parser.add_argument('--k',
                        default=1.0, type=float, help="k as in kx+b")
    parser.add_argument('--bias',
                        default=0.0, type=float, help="b as in kx+b")

    args = parser.parse_args()
    return args

TF          = {'True':True, 'False':False}
args        = param_parse()
dataset     = args.dataname
n_shot      = args.n_shot
n_ways      = args.n_ways
n_queries   = args.n_queries
n_runs      = args.n_runs
be          = args.beta # 0.5 | 1.
l2          = TF[args.norm]  # False | True
cu          = False  # False | True
mth         = args.method
combi       = args.combine
img_size    = args.re_size
flip        = TF[args.flip]
nv          = args.nv
transform   = args.transform
k           = args.k
bias        = args.bias
print(args)

print('>>> Now running {} tasks for {}-way {}-shot {}-query for {} dataset...'.format(n_runs,
                                                                                      n_ways,
                                                                                      n_shot,
                                                                                      n_queries,
                                                                                      dataset))
print('>>> degree:', args.dgr)

# %% ========== ========== ========== ========== ========== ========== ========== ==========
# ---- data loading ----
'''
dataset   = 'miniImagenet' # 'miniImagenet' 'CUB' 'tieredImagenet'
n_shot    = 5 # 1
n_ways    = 5
n_queries = 15
n_runs    = 1000
'''
n_lsamples    = n_ways * n_shot
n_usamples    = n_ways * n_queries
n_samples     = n_lsamples + n_usamples

cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
dgr = args.dgr  # can be "0" "90" "180" "270"
_ = FSLTask.loadDataSet(dataset,
                        dgr=dgr,
                        mth=mth,
                        img_size=img_size,
                        flip=flip,
                        nv=nv)
FSLTask.setRandomStates(cfg)
ndatas, labset, idxset = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg, re_true_lab=True)
ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
labels = torch.arange(n_ways).view(1, 1, n_ways) \
                             .expand(n_runs, n_shot+n_queries, n_ways).clone() \
                             .view(n_runs, n_samples)

max_nsamples = 0

JS = './filelists/'+dataset+'/novel.json' # backup/
with open(JS,"r") as f:
    JSlist = json.load(f)
super_lab = np.array(JSlist['label_names'])

# test-time data augmentation: removed

# %% ========== ========== ========== ========== ========== ========== ========== ==========
# ---------- ----------
#
# now try fewer data
#
# ---------- ----------
tstar     = time.time()

scores    = []
re_dist   = True
bisecting = False
if bisecting:
    dists     = np.zeros((0, n_ways+1), dtype=np.float32)
else:
    dists     = np.zeros((0, n_ways), dtype=np.float32)
re_posi   = False # True | False
if re_posi:
    dists = []

TR = trange(n_runs, leave=True) # former: range(0, n_runs)
for itask in TR:
    i_run  = itask  # 196
    onlyd  = True  # True | False
    onlyda = False  # True | False
    max_ns = max_nsamples  # max_nsamples | 0
    predicts, probs, acc, nu, dataSlabelS = test_DC(i_run, ndatas,
                                                    labels,
                                                    n_lsamples, n_ways, n_shot,
                                                    [], [],  # base_means, base_cov,
                                                    max_nsamples=max_ns,
                                                    beta=0.5, # of no use, we don't test DC any longer
                                                    onlydata=onlyd,
                                                    onlydataA=onlyda,
                                                    TTDA=False,
                                                    multi_query=False)

    support_data, support_label, query_data, query_label = dataSlabelS
    # print([d.shape for d in dataSlabelS])
    n_shot_ = n_shot

    # %% ========== Neighbor evaluation vvv
    acc_nc, nn_pre, dist = NC_o(support_data, support_label, query_data, query_label,
                                n_ways, n_shot_, beta=be, l2=l2, use_mean=True, re_dist=re_dist,
                                bisecting=False, # True,
                                merge_query=0,
                                re_posi=re_posi,
                                transform=transform, k=k, bias=bias)

    if re_dist:
        if re_posi:
            dists.append(dist)
        else:
            dists = np.vstack((dists, dist))
    # %% ========== Neighbor evaluation ^^^

    score = acc_nc[0] * 100
    scores.append(score)
    '''
    print('--- nc {:.3f} {:.3f} {:.3f} ---'.format(*score))
    score = score_ens(scores)
    print('*** nc {:.4f} {:.4f} {:.4f} ***'.format(*score))
    '''
    s_all = score_ens(scores)
    # L   = super_lab[labset[i_run]]
    # print(s_cur, L)
    TR.set_description('{:.2f} | {:.2f}±{:.2f}'.format(score, *s_all))

print('>>> final: {:.2f}±{:.2f}'.format(*s_all))
#%% at last
dgr_dict = {'s2m2':'dgr_val', 'rot':'dgr_rot', 'mix':'dgr_mix'} # NOTE: dgr or dgr_val
dgr_save = dgr_dict[mth]
if re_dist and args.save_dist:
    if transform == 'beta':
        p_bet = '{:.2f}'.format(be)
        p_log = '0'
    elif transform == 'log':
        p_bet = '0'
        p_log = '{:.2f}'.format(bias)
    else:
        print('>>> Error in transform!')
    dis_file = './{}/{}way{}shot/{}/s{}_r{}_f{}_b{}_l{}.plk'.format(dgr_save,
                                                      n_ways,
                                                      n_shot,
                                                      dataset,
                                                      img_size,
                                                      '0' if not dgr else dgr,
                                                      '1' if flip else '0',
                                                      p_bet,
                                                      p_log)
    with open(dis_file, 'wb') as handle:
        pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('--> distances saved at', dis_file)

tnow = time.time()
print('> time to {}:{}'.format('END', (tnow-tstar)/60.))
