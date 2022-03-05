'''
@horsepurve

helpful functions
'''
import json
import logging
import os
import shutil
import numpy as np
import pickle
import copy
import networkx as nx

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
from torch.nn import functional as F

from sklearn.linear_model import LogisticRegression
from evaluate_DC import distribution_calibration

#%%
class FeatDataset(Dataset):  
    def __init__(self, XX, YY):        
        self.x = XX        
        self.y = YY
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return (self.x[idx],
                self.y[idx])
#%%
def _shift(l, A = 0.0):
    _l = copy.deepcopy(l)
    _l[1] += A
    return _l
        
def _diag(l):
    d = np.diag(np.ones(l))
    d = d.flatten()[:-1]
    d = np.insert(d, 0, 0)    
    d = d.reshape((l,l))
    return d+d.T
#%%
def normalize_l2(x, axis=1):
    '''x.shape = (num_samples, feat_dim)'''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 0.0) # 1e-8 | 0.0
    return x
#%%
def train_clf(X_aug, Y_aug, query_data, query_label, 
              n_ways=0, beta=1., l2=False, cuda=False, niter=0):
    '''
    Returns
    -------    
    '''
    if beta != 1.:
        query_data = np.power(query_data[:, ], beta)
    if l2:
        query_data = normalize_l2(query_data)
    
    # simpleshot resnet50 -> change to solver='sag': not good
    classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)
    
    predicts = classifier.predict(query_data)
    probs    = classifier.predict_proba(query_data)
    if n_ways != 0:    
        predicts = np.argmax(probs[:,:n_ways],axis=-1)
        probs    = probs[:,:n_ways] 
        probs    = probs / (np.sum(probs,axis=-1)[:,None])
    acc = np.mean(predicts == query_label)
    nu  = np.sum(predicts == query_label)
    print('acc: {:.3f} ({}/{})'.format(acc, nu, len(predicts)))
    return predicts, probs, acc, nu
#%%
def one_hot_encode(vector):
    n_classes = len(vector.unique())  # 1
    one_hot = torch.zeros((vector.shape[0], n_classes))\
        .type(torch.LongTensor)  # 2
    return one_hot\
        .scatter(1, vector.type(torch.LongTensor).unsqueeze(1), 1)  # 3
#%%
class LgReg(torch.nn.Module):
    '''
    TODO: pre-training
    '''
    def __init__(self, dim, nla):
        super(LgReg, self).__init__()
        self.linear = nn.Linear(dim, nla)
        
    def forward(self, x):
        y_hat = self.linear(x)
        y_hat = torch.sigmoid(y_hat)
        return y_hat
#%%
class LgRegv(torch.nn.Module):
    '''
    TODO: pre-training
    from power to voronoi
    '''
    def __init__(self, dim, nla):
        super(LgRegv, self).__init__()
        self.linear = nn.Linear(dim, nla, bias=False)
        
    def forward(self, x):
        ba = - torch.sum((self.linear.weight/2)**2, dim=1)
        y_hat = self.linear(x) + ba
        y_hat = torch.sigmoid(y_hat)
        return y_hat    
#%%
def give_center(xx, yy):
    nla     = len(set(yy))
    centers = [np.mean(xx[yy==i,:],axis=0) for i in range(nla)]
    centers = np.array(centers)
    return centers

class LgRegvi(torch.nn.Module):
    '''
    TODO: pre-training
    from power to voronoi
    with initialization -- doesn't work
    '''
    def __init__(self, dim, nla, xx, yy):
        super(LgRegvi, self).__init__()
        self.linear = nn.Linear(dim, nla, bias=False)
        cen = give_center(xx, yy)
        self.linear.weight.data = torch.from_numpy(cen*2)
        
    def forward(self, x):
        ba = - torch.sum((self.linear.weight/2)**2, dim=1)
        y_hat = self.linear(x) + ba        
        y_hat = torch.sigmoid(y_hat)         
        return y_hat
#%%
from torch.optim.lr_scheduler import StepLR
def train_clf_torch(X_aug, Y_aug, 
                    query_data, query_label, 
                    n_ways=0, beta=1., 
                    l2=False, cuda=False, 
                    niter=550,
                    prt=True,
                    return_weight=False,
                    opt='adam',
                    lr=0.1):
    '''
    The very basic version.    
    '''
    if cuda:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')

    if l2:
        query_data = normalize_l2(query_data)    
    if beta != 1.:
        query_data = np.power(query_data[:, ], beta)
    
    XX_train = torch.tensor(X_aug, dtype=torch.float32, device=dev) 
    XX_test  = torch.tensor(query_data, dtype=torch.float32, device=dev) 
    YY       = torch.tensor(Y_aug, device=dev)
    
    dim = X_aug.shape[-1]
    nla = len(set(Y_aug))
    
    '''
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, nla)
    )
    '''
    model = LgReg(dim, nla) # LgRegv [Note here]
    # model = LgCenter(dim, nla, X_aug, Y_aug)
    model.to(dev) 
    
    if 'adam' == opt:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, # 0.1, # LR: 0.1 is better than 0.01
            weight_decay=0.01 # LR: 0.01 is better than 0.001
        )    
    elif 'sgd' == opt:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, # 0.001, 
            momentum=0.9, 
            dampening=0.9, 
            weight_decay=0.001
        )
    else:
        print('ERROR Opt.')
        return 'ERROR'
     
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    loss_function = torch.nn.CrossEntropyLoss()
    
    n_iterations = niter
    probs_epo = np.zeros((query_data.shape[0], nla if n_ways == 0 else n_ways))
    probs_i   = 0
    acc_run   = 0   
    
    for i in range(1, n_iterations + 1):
        Z = model(XX_train)
        loss = loss_function(Z, YY)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()        
        
        if i == 1 or i % 25 == 0:
            ## print("Loss at iteration {}: {}".format(i, loss))
            out      = model(XX_test)
            if cuda:
                out  = out.cpu()
            probs    = torch.softmax(out, 1).detach().numpy()
            predicts = np.argmax(probs, axis=-1)
        
            if n_ways != 0:    
                predicts = np.argmax(probs[:,:n_ways],axis=-1)
                probs    = probs[:,:n_ways] 
                probs    = probs / (np.sum(probs,axis=-1)[:,None])
            acc = np.mean(predicts == query_label)
            nu  = np.sum(predicts == query_label)
            # epoch ensemble -----
            if i > 150:
                probs_epo += probs
                probs_i   += 1
                pred_run  = np.argmax(probs_epo, axis=-1)
                acc_run   = np.mean(pred_run == query_label)
            # epoch ensemble -----
            if prt:
                print('{} acc: {:.3f} ({}/{}) | {:.3f}'.format(i,
                                                        acc, 
                                                        nu, 
                                                        len(predicts),
                                                        acc_run))
    if return_weight:
        return predicts, probs, acc, nu, acc_run, \
            model.linear.weight.data.detach().numpy()
    else:
        return predicts, probs, acc, nu, acc_run  
#%%
def w_l(values, v_mean):
    '''
    find weights
    '''
    dis = np.sum((values - v_mean) ** 2, axis=-1)
    S   = np.sum(dis)
    w   = np.log(S / dis)
    return w / np.sum(w)

def td_ens(probs_epo_all, query_label, IT=5):
    '''
    (epoch_candidate, query_samples, prob_dim)
    '''
    shp = probs_epo_all.shape
    print(shp)
    tD = np.zeros((shp[1], shp[2]))
    for s in range(probs_epo_all.shape[1]):
        values = probs_epo_all[:,s,:]
        # plt.plot(values.T)
        v_mean = np.mean(values, axis=0) # 1st estimation
        w      = w_l(values, v_mean)
        v_     = np.dot(values.T, w) # 2nd estimation
        for i in range(IT):
            w  = w_l(values, v_)
            v_ = np.dot(values.T, w)
        tD[s,:] = v_        
    pred_run  = np.argmax(tD, axis=-1) # tD[:,:5]
    acc_run   = np.mean(pred_run == query_label)    
    print('td acc:', acc_run)

def td_mean(a, IT=6):
    '''
    a.shape: (5, 5, 640)
    '''
    a_means = []
    n_ways, n_shot, dime = a.shape
    for i in range(n_ways):
        values = a[i,:,:]
        # plt.plot(values.T)
        v_mean = np.mean(values, axis=0) # 1st estimation
        w      = w_l(values, v_mean)
        v_     = np.dot(values.T, w) # 2nd estimation
        for i in range(IT):
            w  = w_l(values, v_)
            v_ = np.dot(values.T, w)
        a_means.append(v_) 
    a_means = np.array(a_means)
    return a_means
#%%
def train_clf_torch_td (X_aug, Y_aug, 
                    query_data, query_label, 
                    n_ways=0, beta=1., 
                    l2=False, cuda=False, 
                    niter=550):
    '''
    The very basic version 
        + truth discovery among epochs
    '''
    if cuda:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    
    if beta != 1.:
        query_data = np.power(query_data[:, ], beta)
    if l2:
        query_data = normalize_l2(query_data)
    
    XX_train = torch.tensor(X_aug, dtype=torch.float32, device=dev) 
    XX_test  = torch.tensor(query_data, dtype=torch.float32, device=dev) 
    YY       = torch.tensor(Y_aug, device=dev)
    
    dim = X_aug.shape[-1]
    nla = len(set(Y_aug))
    
    '''
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, nla)
    )
    '''
    model = LgReg(dim, nla)
    model.to(dev) 
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.1, 
        weight_decay=0.01
    )
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    n_iterations = niter
    probs_epo = np.zeros((query_data.shape[0], nla if n_ways == 0 else n_ways))
    probs_epo_all = np.zeros((0, query_data.shape[0], nla if n_ways == 0 else n_ways)) # nla
    probs_i   = 0
    acc_run   = 0    
    for i in range(1, n_iterations + 1):
        Z = model(XX_train)
        loss = loss_function(Z, YY)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        
        
        if i == 1 or i % 25 == 0: # %25
            ## print("Loss at iteration {}: {}".format(i, loss))
            out      = model(XX_test)
            if cuda:
                out  = out.cpu()
            probs    = torch.softmax(out, 1).detach().numpy()
            predicts = np.argmax(probs, axis=-1)
        
            if n_ways != 0:    
                predicts = np.argmax(probs[:,:n_ways],axis=-1)
                # probs_bk = probs
                probs    = probs[:,:n_ways] 
                probs    = probs / (np.sum(probs,axis=-1)[:,None])
            acc = np.mean(predicts == query_label)
            nu  = np.sum(predicts == query_label)
            # epoch ensemble -----
            if i > 150:
                # --- stack all
                probs_epo_all = np.concatenate((probs_epo_all,
                                                np.expand_dims(probs, axis=0))) # probs_bk
                # ---
                probs_epo += probs
                probs_i   += 1
                pred_run  = np.argmax(probs_epo, axis=-1)
                acc_run   = np.mean(pred_run == query_label)
            # epoch ensemble -----
            print('acc: {:.3f} ({}/{}) | {:.3f}'.format(acc, 
                                                        nu, 
                                                        len(predicts),
                                                        acc_run))
    return predicts, probs, acc, nu, probs_epo_all  
#%%
def train_clf_torchb(X_aug, Y_aug, 
                     query_data, query_label, 
                     model=0, bcs=128,
                     n_ways=0, beta=1., 
                     l2=False, cuda=False, 
                     niter=550,
                     seeiter=25,ens_cut=150,
                     optmz='sgd',lr=0.01,
                     prt=True):
    
    '''
    In this version we add batch size
    model == 0 means no pretraining
    seeiter: epoch interval for print acc
    optmz: SGD or Adam
    n_ways == 0 means we don't expand few dataset
    ens_cut: when to start ensemble
    '''
    
    if cuda:
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    
    if beta != 1.:
        query_data = np.power(query_data[:, ], beta)
    if l2:
        query_data = normalize_l2(query_data)
    
    XX_train = torch.tensor(X_aug, dtype=torch.float32, device=dev) 
    XX_test  = torch.tensor(query_data, dtype=torch.float32, device=dev) 
    YY       = torch.tensor(Y_aug, device=dev)
    
    dim = X_aug.shape[-1]
    nla = len(set(Y_aug))
    
    '''
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, nla)
    )
    '''
    if 0 == model:
        model = LgReg(dim, nla) # LgRegv
        # model = LgRegvi(dim, nla, X_aug, Y_aug) # LgRegvi | LgCenter
    model.to(dev) 
    
    if 'sgd' == optmz:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, # 0.1, 
            momentum=0.9,
            dampening=0.9,             
            weight_decay=0.001
        )
    elif 'adam' == optmz:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, # 0.01,
            weight_decay=0.01
        )
    else:
        optimizer = None
        print('Unknown Optimizer!')
    # scheduler = StepLR(optimizer, step_size=75, gamma=0.1)
    
    loss_function = torch.nn.CrossEntropyLoss()
    
    # ----------
    dataset_train = FeatDataset(XX_train, YY)    
    train_iter    = torch.utils.data.DataLoader(dataset_train, 
                                                batch_size=bcs, 
                                                shuffle=True, # True | False
                                                num_workers=0)    
    # DA, LA = iter(train_iter).next() # have a look
    # ----------
    
    n_iterations = niter
    probs_epo = np.zeros((query_data.shape[0], nla if n_ways == 0 else n_ways))
    probs_i   = 0
    acc_run   = 0    
    for i in range(1, n_iterations + 1):
        for DA, LA in train_iter:
            Z = model(DA)
            loss = loss_function(Z, LA)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()        
        
        if i == 1 or i % seeiter == 0:
            ## print("Loss at iteration {}: {}".format(i, loss))
            out      = model(XX_test)
            if cuda:
                out  = out.cpu()
            probs    = torch.softmax(out, 1).detach().numpy()
            predicts = np.argmax(probs, axis=-1)
        
            if n_ways != 0:    
                predicts = np.argmax(probs[:,:n_ways],axis=-1)
                probs    = probs[:,:n_ways] 
                probs    = probs / (np.sum(probs,axis=-1)[:,None])
            acc = np.mean(predicts == query_label)
            nu  = np.sum(predicts == query_label)
            # epoch ensemble -----
            if i > ens_cut:
                probs_epo += probs
                probs_i   += 1
                pred_run  = np.argmax(probs_epo, axis=-1)
                acc_run   = np.mean(pred_run == query_label)
            # epoch ensemble -----
            if prt:
                print('acc: {:.3f} ({}/{}) | {:.3f}'.format(acc, 
                                                        nu, 
                                                        len(predicts),
                                                        acc_run))
    return predicts, probs, acc, nu, model, acc_run 

#%%    
def get_task(i, ndatas, labels, n_lsamples):
    ''' 
    get one task
    '''
    i_run = i
    
    support_data  = ndatas[i][:n_lsamples].numpy() # (25, 640)
    support_label = labels[i][:n_lsamples].numpy()
    query_data    = ndatas[i][n_lsamples:].numpy() # (75, 640)
    query_label   = labels[i][n_lsamples:].numpy()
    
    dataSlabelS = [copy.deepcopy(support_data), 
                   copy.deepcopy(support_label),
                   copy.deepcopy(query_data), 
                   copy.deepcopy(query_label)]
    
    return dataSlabelS    
#%%
def test_DC(i, ndatas, labels,
            n_lsamples,
            n_ways,
            n_shot,
            base_means,
            base_cov,
            max_nsamples = 0,
            beta         = 0.5,
            onlydata     = False,
            onlydataA    = False,
            TTDA         = False,
            multi_query  = False):
    '''    
    Parameters
    ----------
    i           : no. task
    onlydata    : *only* return the support & query data
    onlydataA   : *only* return the augmented data
    samples_ept : how many samples to sample for each support point?
    n_lsamples  : how many support samples in total?
    
    TTDA        : test-time data augmentation
    multi_query : also rotate the queries, or not?
    Returns
    -------    
    '''
    # i = 945
    i_run = i
    
    # ***** TTDA *****
    if TTDA:
        # e.g.: ndatasS shape (2, 10, 100, 640)    
        support_data  = ndatas[0][i][:n_lsamples] # (25, 640)
        support_label = labels   [i][:n_lsamples].numpy()
        query_data    = ndatas[0][i][n_lsamples:] # (75, 640)
        query_label   = labels   [i][n_lsamples:].numpy()
        for ro in range(1,ndatas.shape[0]):
            s_data  = ndatas[ro][i][:n_lsamples] # (25, 640)
            s_label = labels    [i][:n_lsamples].numpy()
            support_data  = np.vstack((support_data, s_data))
            support_label = np.hstack((support_label, s_label))    
            if multi_query:
                q_data  = ndatas[ro][i][n_lsamples:] # (25, 640)
                q_label = labels    [i][n_lsamples:].numpy()
                query_data  = np.vstack((query_data, q_data))
                query_label = np.hstack((query_label, q_label))                
    # ***** TTDA *****
    
    else:
        support_data  = ndatas[i][:n_lsamples].numpy() # (25, 640)
        support_label = labels[i][:n_lsamples].numpy()
        query_data    = ndatas[i][n_lsamples:].numpy() # (75, 640)
        query_label   = labels[i][n_lsamples:].numpy()
    
    dataSlabelS = [copy.deepcopy(support_data), 
                   copy.deepcopy(support_label),
                   copy.deepcopy(query_data), 
                   copy.deepcopy(query_label)]
    
    if onlydata:
        return 0, 0, 0, 0, dataSlabelS
    
    # ---- Tukey's transform
    # beta = 0.5
    support_data = np.power(support_data[:, ] ,beta)
    query_data   = np.power(query_data[:, ] ,beta)
    
    # ---- distribution calibration and feature sampling
    sampled_data  = []
    sampled_label = []
    if 0 == max_nsamples:
        num_sampled = int(750/n_shot)
    else:
        num_sampled = int(max_nsamples * n_ways / n_lsamples)
    # num_sampled=1
    for i in range(n_lsamples):
        mean, cov = distribution_calibration(support_data[i], 
                                             base_means, 
                                             base_cov, k=2)
        # sampled_data.append(mean)
        sampled_data.append(np.random.multivariate_normal(mean=mean, 
                                                          cov=cov, 
                                                          size=num_sampled))
        sampled_label.extend([support_label[i]]*num_sampled)
    
    sampled_data = np.concatenate([sampled_data[:]]).reshape(n_ways * n_shot * num_sampled, -1)
    # sampled_data: (3750, 640) w/ corresponding labels
    if onlydataA:
        return 0, 0, 0, \
               (sampled_data, np.array(sampled_label)), dataSlabelS
        
    X_aug = np.concatenate([support_data, sampled_data])
    Y_aug = np.concatenate([support_label, sampled_label])
    
    # ---- train classifier
    predicts, probs, acc, nu = train_clf(X_aug, Y_aug, 
                                         query_data, query_label)
    # acc_list.append(acc)
    # print('{}, {:.3f}, {:.3f}'.format(i_run,acc,float(np.mean(acc_list))), flush=True)
    # print('DC acc for {}:\n{}'.format(i_run, acc))
    return predicts, probs, acc, nu, dataSlabelS
#%%
def only_few_data(support_data, support_label, beta=1., l2=False):
    xx = copy.deepcopy(support_data)
    yy = copy.deepcopy(support_label)
    if l2:
        xx = normalize_l2(xx)    
    if beta != 1.:
        xx = np.power(xx[:, ] ,beta)
    return xx, yy
#%%
def fill_base_data(xx, yy, C_base, C_novel, n_few, dd, max_nsamples, 
                   beta=1., l2=False):
    '''
    Parameters
    ----------
    xx : TYPE
        DESCRIPTION.
    yy : TYPE
        DESCRIPTION.
    C_base : TYPE
        DESCRIPTION.
    C_novel : TYPE
        DESCRIPTION.
    n_few : TYPE
        DESCRIPTION.
    dd : TYPE
        DESCRIPTION.
    max_nsamples : TYPE
        600 for miniImagenet
        60  for CUB
    beta : TYPE, optional
        DESCRIPTION. The default is 1..
    l2 : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    xx : TYPE
        DESCRIPTION.
    yy : TYPE
        DESCRIPTION.
    '''
    for i in range(C_base):
        ind   = np.random.choice(max_nsamples, n_few, replace=False)  
        added = dd[i, ind, :]
        if beta != 1.:            
            added = np.power(added, beta)
        if l2:
            added = normalize_l2(added)
        xx = np.vstack((xx,added))
        yy = np.append(yy,[C_novel+i] * n_few)
    print('data shape:', xx.shape)
    print('label shape:', yy.shape)
    return xx, yy
#%%
from numpy import linalg as LA
fun = lambda a,b : LA.norm(a[:,None,:]-b[None,:,:],axis=2)

def NC_bk(support_data, support_label, 
       query_data, query_label, 
       n_ways, n_shot,
       beta=1., l2=False,
       use_mean=False):
    '''
    nearest cluster classification
    use_mean: use geometric mean shot
    bk: 1st successful version backup
    '''
    n_allq  = query_data.shape[0]
    dime    = support_data.shape[-1]
    
    a = copy.deepcopy(support_data)
    b = copy.deepcopy(query_data)
    
    if l2:
        a = normalize_l2(a)
        b = normalize_l2(b)
    if beta != 1.:
        a = np.power(a[:, ] ,beta)
        b = np.power(b[:, ] ,beta)
    if use_mean:
        a = redata(a, n_ways, n_shot)
        a = a.reshape(n_ways, n_shot, dime)
        a = np.mean(a, axis=1) # "a = td_mean(a)" is bad!
    
    dist   = fun(b, a) # do distance
    
    accl = []
    # 1-NN
    min_id = np.argmin(dist,axis=-1)
    nn_pre = support_label[min_id]
    nn_acc = np.mean(nn_pre == query_label)
    print('1-NN: {:.4f}'.format(nn_acc))
    if use_mean:
        return [nn_acc, 0, 0]
    accl.append(nn_acc)

    # NC
    poer = 2
    dista  = 1./(dist.reshape((n_allq, n_shot, n_ways))**poer)
    distc  = np.sum(dista, axis=1) # clsuter-level distance
    min_id = np.argmax(distc,axis=-1)
    nc_pre = support_label[min_id]
    nc_acc = np.mean(nc_pre == query_label)
    print('1-NC (1/d): {:.4f}'.format(nc_acc))
    accl.append(nc_acc)
    
    # NC
    dista  = dist.reshape((n_allq, n_shot, n_ways))    
    distc  = np.sum(dista, axis=1) # clsuter-level distance
    min_id = np.argmin(distc,axis=-1)
    nc_pre = support_label[min_id]
    nc_acc = np.mean(nc_pre == query_label)
    print('1-NC (d): {:.4f}'.format(nc_acc))
    accl.append(nc_acc)
    
    return accl
#%%
def remove_far(support_data, support_label,
               n_ways, n_shot,
               remove_i=1):
    '''  
    e.g. remove 1 farthest from 5
    n_shot = 5
    '''
    a = copy.deepcopy(support_data)
    a = redata(a, n_ways, n_shot)
    a = a.reshape(n_ways, n_shot, -1)
    l = np.array(relabel(support_label, n_ways, n_shot))
    l = l.reshape((n_ways, n_shot))
    a_mean = np.mean(a, axis=1) # "a = td_mean(a)" is bad!    
    dist   = a - 0 # a_mean
    dist   = np.sum(dist**2, axis=-1)
    disti  = np.argsort(dist, axis=-1)
    if remove_i != 0:
        disti  = disti[:,:-remove_i]
    aa = [a[i][disti[i]] for i in range(n_ways)]
    aa = np.array(aa)
    n_shot = n_shot - remove_i
    aa = aa.transpose((1,0,2)).reshape((n_ways*n_shot, -1))
    ''' check it
    np.sum(support_data,axis=-1)
    np.sum(aa,axis=-1)
    '''
    return aa, n_shot
#%%
def NC(support_data, support_label, 
       query_data, query_label, 
       n_ways, n_shot,
       beta=1., l2=False,
       use_mean=False,
       re_dist=False,
       bisecting=False,
       merge_query=0):
    '''
    nearest cluster classification
    use_mean: use geometric mean shot
    '''
    n_allq  = query_data.shape[0]
    dime    = support_data.shape[-1]
    
    a = copy.deepcopy(support_data)
    b = copy.deepcopy(query_data)
    
    ##### not good
    # a, n_shot = remove_far(a, support_label, n_ways, n_shot, remove_i=1)
    
    if l2:
        a = normalize_l2(a)
        b = normalize_l2(b)
    if beta != 1.:
        a = np.power(a[:, ] ,beta)
        b = np.power(b[:, ] ,beta)

        #a = np.log(a+0.04)
        #b = np.log(b+0.04)
    if use_mean:
        a = redata(a, n_ways, n_shot) # relabel(support_label, n_ways, n_shot)
        a = a.reshape(n_ways, n_shot, dime)
        a = np.mean(a, axis=1) # "a = td_mean(a)" is bad!
    
    ''' for manual ensemble: transfor only query
    b = normalize_l2(b)
    b = np.power(b[:, ] ,0.5)
    '''
    
    dist   = fun(b, a) # do distance
    if merge_query != 0:
        dist = dist.reshape((merge_query,-1,5))
        dist = np.mean(dist, axis=0)
        query_label = query_label.reshape((merge_query,-1))[0]
    
    accl = []
    # 1-NN
    min_id = np.argmin(dist,axis=-1)
    nn_pre = support_label[min_id]
    nn_acc = np.mean(nn_pre == query_label)
    ## print('1-NN: {:.4f}'.format(nn_acc))
    if use_mean:
        if re_dist:
            return [nn_acc, 0, 0], nn_pre, dist
        else:
            return [nn_acc, 0, 0], nn_pre
    accl.append(nn_acc)

    # NC
    poer = 2
    dista  = 1./(dist.reshape((n_allq, n_shot, n_ways))**poer)
    distc  = np.sum(dista, axis=1) # clsuter-level distance
    min_id = np.argmax(distc,axis=-1)
    nc_pre = support_label[min_id]
    nc_acc = np.mean(nc_pre == query_label)
    print('1-NC (1/d): {:.4f}'.format(nc_acc))
    accl.append(nc_acc)
    
    # NC
    dista  = dist.reshape((n_allq, n_shot, n_ways))    
    distc  = np.sum(dista, axis=1) # clsuter-level distance
    min_id = np.argmin(distc,axis=-1)
    nc_pre = support_label[min_id]
    nc_acc = np.mean(nc_pre == query_label)
    print('1-NC (d): {:.4f}'.format(nc_acc))
    accl.append(nc_acc)
    
    return accl
#%%
def NC_o(support_data, support_label, 
         query_data, query_label, 
         n_ways, n_shot,
         beta        = 1., 
         l2          = False,
         use_mean    = False,
         re_dist     = False,
         bisecting   = False,
         merge_query = 0,
         re_posi     = False,
         transform   = 'beta', # beta | log
         k           = 1., # as in kx + b
         bias        = 0.
         ):
    '''
    nearest cluster classification
    use_mean: use geometric mean shot
    NC_o: this version we *only* use mean
    
    beta     = 0.5
    l2       = True
    use_mean = True
    '''
    n_allq  = query_data.shape[0]
    dime    = support_data.shape[-1]
    
    a = copy.deepcopy(support_data)
    b = copy.deepcopy(query_data)    
    
    if l2: # do l2-norm firstly
        a = normalize_l2(a)
        b = normalize_l2(b)
    
    a = k * a + bias
    b = k * b + bias
    
    if beta != 1.:
        if transform == 'beta':
            a = np.power(a[:, ] ,beta)
            b = np.power(b[:, ] ,beta)
        elif transform == 'log':
            a = np.log(a) # np.log(a+0.04)
            b = np.log(b) # np.log(b+0.04)        
    
    if use_mean:
        a = redata(a, n_ways, n_shot) # relabel(support_label, n_ways, n_shot)
        a = a.reshape(n_ways, n_shot, dime)        
        a = np.mean(a, axis=1) # "a = td_mean(a)" is bad!    
    
    dist   = fun(b, a) # do distance
    
    # 1-NN
    if not bisecting:
        min_id = np.argmin(dist,axis=-1)
        nn_pre = support_label[min_id]
        nn_acc = np.mean(nn_pre == query_label)
        ## print('1-NN: {:.4f}'.format(nn_acc))
        if use_mean:
            if re_dist:
                if re_posi:
                    return [nn_acc, 0, 0], nn_pre, [a, b]
                else:
                    return [nn_acc, 0, 0], nn_pre, dist                
            else:
                return [nn_acc, 0, 0], nn_pre
    else:
        id_sort = np.argsort(dist,axis=-1)
        min_id1 = id_sort[:,0]
        min_id2 = id_sort[:,1]
        nn_pre  = support_label[min_id1]
        nn_acc  = np.mean(nn_pre == query_label)
        x1 = a[min_id1]
        x2 = a[min_id2]
        w  = x1 - x2
        bi = 0.5 * (LA.norm(x1,axis=1)**2 - LA.norm(x2,axis=1)**2)
        d  = np.abs(np.einsum('ij, ij->i', (x1-x2), b) - bi) / LA.norm(x1-x2,axis=1)
        return [nn_acc, 0, 0], nn_pre, np.hstack((dist, d[:,None]))
#%%
from scipy.stats import mode

def metric_class_type(gallery, 
                      query, 
                      train_label, 
                      test_label, 
                      n_ways, n_shot, nNei=1,
                      train_mean=None, 
                      norm_type='CL2N'):
    ''' 
    gallery = train_data
    query   = test_data
    '''
    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]
    gallery = gallery.reshape(n_ways, 
                              n_shot, 
                              gallery.shape[-1]).mean(1)
    train_label = train_label[::n_shot]
    subtract = gallery[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1)
    idx = np.argpartition(distance, nNei, axis=0)[:nNei]
    nearest_samples = np.take(train_label, idx)
    out = mode(nearest_samples, axis=0)[0]
    out = out.astype(int)
    test_label = np.array(test_label)
    acc = (out == test_label).mean()
    return acc
#%%
relabel = lambda support_label, n_ways, n_smp:  \
    support_label.reshape((n_smp, n_ways)).T.flatten().tolist()
redata  = lambda support_data, n_ways, n_smp: \
    support_data.reshape((n_smp, n_ways, support_data.shape[-1])).transpose(1,0,2).reshape((n_ways*n_smp, support_data.shape[-1]))
#%%
def simple_shot(support_data, support_label, 
                query_data, query_label, 
                n_ways, n_shot, n_queries,
                out_mean_
                ):
    ''' 
    '''
    train_label = relabel(support_label, n_ways, n_shot)
    test_label  = relabel(query_label,   n_ways, n_queries)
    train_data  = redata(support_data,   n_ways, n_shot)
    test_data   = redata(query_data,     n_ways, n_queries)
    
    acc_s = []
    acc = metric_class_type(train_data, 
                            test_data, 
                            train_label, 
                            test_label, 
                            n_ways, n_shot, 
                            train_mean=out_mean_,
                            norm_type='UN') 
    acc_s.append(acc)
    acc = metric_class_type(train_data, 
                            test_data, 
                            train_label, 
                            test_label, 
                            n_ways, n_shot, 
                            train_mean=out_mean_,
                            norm_type='L2N') 
    acc_s.append(acc)
    acc = metric_class_type(train_data, 
                            test_data, 
                            train_label, 
                            test_label, 
                            n_ways, n_shot, 
                            train_mean=out_mean_,
                            norm_type='CL2N') 
    acc_s.append(acc)
    
    return acc_s
#%%
def see_rlt(i_run,resu,resu0,resu1,resu2,
            resu0_, resu1_):
    print('#{}:\nDC - {:.3f}\nfew - {:.3f}\nfew+(5) - {:.3f}\nfew+(15) - {:.3f}'.format(i_run, 
                            resu [i_run][2],                                                   
                            resu0[i_run][2],
                            resu1[i_run][2],
                            resu2[i_run][2]))   
    if len(resu0_) != 0:
        print('PY: {:.3f} {:.3f}'.format(resu0_[i_run][2], 
                                         resu1_[i_run][2]))
    
    return resu [i_run][2], \
           resu0[i_run][2], \
           resu1[i_run][2], \
           resu2[i_run][2]

#%%
def do_mean(support_data, support_label, n_ways,
            mth=''):
    ''' 
    mth: '' or 'td'
    '''
    if mth == '':
        means = [np.mean(support_data[support_label==i],
                         axis=0) for i in range(n_ways)]
        means = np.array(means)
        return means
    elif mth == 'td':
        a_means = []
        IT = 3
        for i in range(n_ways):
            values = support_data[support_label==i]
            v_mean = np.mean(values, axis=0) # 1st estimation
            w      = w_l(values, v_mean)
            v_     = np.dot(values.T, w) # 2nd estimation
            for j in range(IT):
                w  = w_l(values, v_)
                v_ = np.dot(values.T, w)
            a_means.append(v_) 
        a_means = np.array(a_means)
        return a_means
    else:
        print('ERROR method')
        return 'ERROR method!'

def NN(a, b, 
       support_label, query_label,
       n_ways, n_shot,
       mth='mean',
       provided_mean=[],
       re_dist=False,
       re_mean=False,
       re_a   =False, 
       prt=True):
    ''' 
    nearest neighbor-like: center, CIVD, etc.
    '''
    aa = copy.deepcopy(a)
    if 'mean' == mth:
        '''
        cluster -> get mean -> do NN
        '''
        if 0 == len(provided_mean):
            a = do_mean(a, support_label, n_ways)
        else:
            a = provided_mean
        dist = fun(b, a) # do distance
        min_id = np.argmin(dist,axis=-1)
        nn_pre = np.arange(n_ways)[min_id] # support_label[min_id]
        nn_acc = np.mean(nn_pre == query_label)
        if prt:
            print('mean NN: {:.4f}'.format(nn_acc))
    elif '1od' == mth:
        '''
        1-over-distance, i.e., IVD
        NOTE: this one does NOT work for iterative manner!
        '''
        dist   = fun(b, a) # do distance
        poer   = 2
        dist   = 1./(dist**poer)
        distc  = do_mean(dist.T, support_label, n_ways)
        min_id = np.argmax(distc,axis=0).T
        nn_pre = support_label[min_id]
        nn_acc = np.mean(nn_pre == query_label)
        print('1-NC (1/d): {:.4f}'.format(nn_acc))
    else:
        print('ERROR method')
        return 'ERROR method'
    
    if re_dist:
        if re_mean:
            return nn_pre, nn_acc, dist, a
        elif re_a:
            return nn_pre, nn_acc, dist, fun(aa, a)
        else:
            return nn_pre, nn_acc, dist
    
    return nn_pre, nn_acc
#%%
def NCs(support_data, support_label, 
        query_data, query_label, 
        n_ways, n_shot,
        beta=1., l2=False,
        IT=3):
    '''
    nearest cluster classification
    use_mean: use geometric mean shot
    NCs: transductive, or, iterative manner
    IT: number of iterations
    '''    
    a = copy.deepcopy(support_data)
    b = copy.deepcopy(query_data)
    
    # ~~~ transformation ~~~
    if l2:
        a = normalize_l2(a)
        b = normalize_l2(b)
    if beta != 1.:
        a = np.power(a[:, ] ,beta)
        b = np.power(b[:, ] ,beta)

        #a = np.log(a+0.04)
        #b = np.log(b+0.04)
    # ~~~ transformation ~~~
    
    accs = []
    mth  = 'mean' 
    nn_pre, nn_acc = NN(a, b, support_label, query_label, 
                        n_ways, n_shot, mth=mth)
    accs.append(nn_acc)
    for i in range(IT):
        trans_mean = do_mean(np.vstack((a,b)),
                             np.concatenate((support_label, nn_pre)),
                             n_ways)
        
        nn_pre, nn_acc = NN(a, b, support_label, query_label, 
                            n_ways, n_shot, mth=mth,
                            provided_mean=trans_mean)
        accs.append(nn_acc)
    return nn_pre, accs
#%%
'''
color scheme --

plt.rcParams["image.cmap"] = "Set1"
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

ref --
https://stackoverflow.com/questions/46768859/how-to-change-the-default-colors-for-multiple-plots-in-matplotlib
https://matplotlib.org/stable/tutorials/colors/colormaps.html
'''
#%%
def transf(a, 
           transform = 'beta', # beta or log         
           beta=0.5, 
           l2=True,
           k=1.0,
           bias=0.0):
    ''' 
    possible methods: beta, l2
    '''
    # 1st, l2norm
    if l2:
        a = normalize_l2(a)
    
    # 2nd, linear
    a = k * a + bias
    
    # 3rd, beta or log
    if beta != 1.:
        if transform == 'beta':
            a = np.power(a[:, ] ,beta)
        elif transform == 'log':
            a = np.log(a) # np.log(a+0.04)
    return a
#%%
def transf_bk(a, beta=0.5, l2=True):
    ''' 
    possible methods: beta, l2
    '''
    if l2:
        a = normalize_l2(a)
    if beta != 1.:
        a = np.power(a[:, ] ,beta)

        # a = np.log(a+0.04)
    return a
#%%
def fewer(support_data , support_label, 
          n_ways,
          big_shot, small_shot,
          scores):
    ''' 
    few to fewer, e.g.
    big_shot   = 50
    small_shot = 5
    '''
    selct_id  = []
    selct_la  = []
    for c in range(n_ways):
        Fs     = np.array([False]*big_shot)
        ix     = np.random.choice(big_shot, small_shot, replace=False)
        Fs[ix] = True
        selct_id.append(Fs)
        selct_la.append([c]*small_shot)
    selct_id  = np.array(selct_id).flatten()
    selct_la  = np.array(selct_la).flatten()
    flt_data  = redata(support_data, n_ways, big_shot)
    # it's now like:
    # print(relabel(support_label, n_ways, big_shot))
    # then we reshape againdef score_ens(scores):
    scores = np.array(scores)
    return np.mean(scores, axis=0)

    supp_la = relabel(selct_la,           n_ways, small_shot)
    supp_da = redata (flt_data[selct_id], n_ways, small_shot)
    return supp_da, np.array(supp_la)
#%%
def load_pkl(fname):
    with open(fname, 'rb') as handle:
        dists = pickle.load(handle)
    return dists

def dist_ens(distsS):
    # distsS = [dists, dists90, dists180, dists270]
    distsS = np.stack(distsS)
    # distsS = 1/distsS
    distsS = -distsS
    distsE = np.mean(distsS, axis=0)
    
    return distsE

def checkAacc_bk(distsE, query_label):
    min_id = np.argmax(distsE,axis=-1)
    l_long = np.array(query_label.tolist()*1000)
    print(np.mean(min_id==l_long))    

def checkAacc(distsE, query_label, n_runs=2000):
    N = n_runs
    min_id = np.argmax(distsE,axis=-1)
    l_long = np.array(query_label.tolist()*N) # Note: use 2000
    TF     = min_id==l_long
    TF     = TF.reshape((N, -1))
    score  = np.mean(TF, axis=1) * 100
    mean, width = score_ens(score)
    print('%4.2f±%4.2f'%(mean, width))
    return mean, width

def or_recursive(*l):
    if len(l) == 1:
        return l[0].astype(bool)
    elif len(l) == 2:
        return np.logical_or(l[0],l[1])
    elif len(l) > 2:
        return or_recursive(or_recursive(*l[:2]),or_recursive(*l[2:]))

def merge_count(id_all, l_long):
    # usage:
    # values, counts = np.unique(all_id[:,0], return_counts=True)
    max_ids = []
    for i in range(len(l_long)):
        values, counts = np.unique(id_all[:,i], return_counts=True)
        max_id = values[np.argmax(counts)]
        max_ids.append(max_id)
    return np.array(max_ids)
#% %
def score_ens_bk(scores):
    scores = np.array(scores)
    return np.mean(scores, axis=0)
#% %
def score_ens(scores): 
    iter_num = len(scores)
    score = np.array(scores)
    mean   = np.mean(score)
    std    = np.std(score)
    interv = 1.96* std/np.sqrt(iter_num)
    return mean, 1.96*std/np.sqrt(iter_num)
#%%
def name_exp(dataset, n_ways, n_shot, 
             img_size, dgr, flip,
             transform, be, bias,
             k = 0, w = 0, pre = '', backbone=''):
    '''
    load one experiment from arguments
    '''
    # pre = 'dgr_val' or 'dgr'
    if pre == '':
        print('>>>dir ERROR')
        return 'ERROR'
    if transform == 'beta':
        p_bet = '{:.2f}'.format(be)
        p_log = '0'
    elif transform == 'log':
        p_bet = '0'
        p_log = '{:.2f}'.format(bias)
    else:
        print('>>> Error in transform!')
    if k == 0:
        dis_file = './{}/{}way{}shot/{}/s{}_r{}_f{}_b{}_l{}.plk'.format(pre,
                                                n_ways, 
                                                n_shot, 
                                                dataset, 
                                                img_size, 
                                                dgr, # '0' if not dgr else dgr, 
                                                '1' if flip else '0',
                                                p_bet,
                                                p_log)    
    else:
        p_w = '{:.1f}'.format(w)
        dis_file = './{}/{}way{}shot/{}/s{}_r{}_f{}_b{}_l{}_k{}_w{}.plk'.format(pre,
                                                n_ways, 
                                                n_shot, 
                                                dataset, 
                                                img_size, 
                                                dgr, # '0' if not dgr else dgr, 
                                                '1' if flip else '0',
                                                p_bet,
                                                p_log,
                                                k,
                                                p_w)    
    if backbone != '':
        return dis_file.replace(dataset, dataset+'/'+backbone)
    return dis_file

class Run:
    def __init__(self, 
                 dataset,
                 n_shot, 
                 n_ways,
                 img_size,
                 dgr,
                 flip,
                 transform,
                 be,
                 bias,
                 k=0,
                 w=0,
                 check=True,
                 pre='',
                 backbone='',
                 n_runs=2000):
        # pre = 'dgr_val' or 'dgr'
        if pre == '':
            print('>>>dir ERROR')
            return 'ERROR'        
        # run params   
        self.dataset = dataset     #= 'miniImagenet'
        self.n_shot = n_shot       #= 5 # 1 5
        self.n_ways = n_ways       #= 5
        self.img_size = img_size   #= 84
        self.dgr = dgr             #= '0'
        self.flip = flip           #= False
        self.transform = transform #= 'beta' # beta log
        self.be = be               #= 0.75
        self.bias = bias           #= 0.0
        self.k = k                 #= 9
        self.w = w                 #= 4
        
        # fixed params
        self.N = n_runs
        self.n_queries = 15
        self.query_label = np.arange(n_ways).tolist() * self.n_queries
        self.query_label = np.array(self.query_label)
        self.Ql = relabel(self.query_label, n_ways, self.n_queries)
        self.Ql = np.array(self.Ql)
        
        # load run
        self.expf = name_exp(self.dataset, self.n_ways, self.n_shot, 
                             self.img_size, self.dgr, self.flip, 
                             self.transform, self.be, self.bias,
                             k = self.k, w = self.w, pre=pre, backbone=backbone)
        print('> loading:', self.expf)
        self.dists = load_pkl(self.expf) 
        
        if check:
            self.check()
        
    def check(self):
        if self.k == 0:
            self.mean, self.width = checkAacc(-self.dists, self.query_label, n_runs=self.N)
        else:
            self.mean, self.width = checkAacc(-self.dists, self.Ql, n_runs=self.N)
    
#%%
def pull_to_base(S, out_mean_s, R):
    '''
    using R nearest base classes to pull it
    '''
    distS = fun(S, out_mean_s)
    fetched_base = np.argsort(distS,axis=-1)[:,:R]
    _part2 = out_mean_s[fetched_base]
    _d = _part2 - S[:,None,:]
    _d = np.linalg.norm(_d, axis=-1, keepdims=True)
    alpha = 2
    wei = 1/(1 + _d**alpha)
    return (S + np.sum(_part2*wei,axis=1)) / (1 + np.sum(wei,axis=1))
    # return (S+_part2) / (1+R)

#%%
def push_from_base(S, out_mean_s, R):
    '''
    using R nearest base classes to pull it
    '''
    distS = fun(S, out_mean_s)
    fetched_base = np.argsort(distS,axis=-1)[:,:R]
    _part2 = out_mean_s[fetched_base]
    _d = _part2 - S[:,None,:]
    _d = np.linalg.norm(_d, axis=-1, keepdims=True)
    alpha = 1
    wei = 1/(1 + _d**alpha)
    # return (S + np.sum(_part2*wei,axis=1)) / (1 + np.sum(wei,axis=1))
    R_ = _part2.shape[1] + 1
    A  = 10.
    return S * (2-1/R_) - np.sum(_part2*wei,axis=1) * (1/A)
    # return (S+_part2) / (1+R)

#%%
#%%
#%%
#%%

#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ========== ========== ==========
