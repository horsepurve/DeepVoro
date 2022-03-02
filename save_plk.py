from __future__ import print_function

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import csv
import os
import collections
import pickle
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from io_utils import model_dict, parse_args
from data.datamgr import SimpleDataManager , SetDataManager
import configs
import copy
from methods.baselinetrain import BaselineTrain

import wrn_mixup_model # wrn_model

import torch.nn.functional as F

from io_utils import parse_args, get_resume_file ,get_assigned_file
from os import path

use_gpu = torch.cuda.is_available()

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 
    def forward(self, x):
        return self.module(x)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def extract_feature(val_loader, model, checkpoint_dir, 
                    dataname, dgr="0",
                    tag='last',set='base',
                    redo=False,
                    flip=False,
                    img_size=84):
    '''
    dgr = "0" # "0" | 90 | 180 | 270
    redo: generate feature again
    '''
    save_dir          = '{}/{}'.format(checkpoint_dir, tag)
    # feat_file_name  = '{}/{}_features.plk{}'.format(save_dir, set, dgr) 
    feat_file_name    = '{}/{}_features_s{}_r{}_f{}.plk'.format(save_dir, 
                                                                set, 
                                                                img_size, 
                                                                '0' if not dgr else dgr, 
                                                                '1' if flip else '0') 
    print('>>> writing to {} for {} ...'.format(feat_file_name, dataname))
    if os.path.isfile(feat_file_name):
        if redo:
            print('>>> found old features, re-generating...')    
        else:
            data = load_pickle(feat_file_name)
            return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    #model.eval()
    with torch.no_grad():
        
        output_dict = collections.defaultdict(list)

        for i, (inputs, labels) in enumerate(val_loader):
            print(i, '-', end='', flush=True)            
            # ~~~~~ rotation ~~~~~ 
            # TODO: fix a bug about dgr '0'
            if dgr == "0" and flip:
                inputs = inputs.flip(2) # 0 degree, flip it
            if dgr == "90":
                if flip:
                    inputs = inputs.permute(0,1,3,2) # 90 degree, flip it
                else:
                    inputs = inputs.permute(0,1,3,2).flip(2) # 90 degree
            elif dgr == "180":
                inputs = inputs.permute(0,1,3,2).flip(2) # 90 degree
                if flip:
                    inputs = inputs.permute(0,1,3,2) # 180 degree, flip it
                else:
                    inputs = inputs.permute(0,1,3,2).flip(2) # 180 degree
            elif dgr == "270":
                inputs = inputs.permute(0,1,3,2).flip(2) # 90 degree
                inputs = inputs.permute(0,1,3,2).flip(2) # 180 degree
                if flip:
                    inputs = inputs.permute(0,1,3,2) # 270 degree, flip it
                else:
                    inputs = inputs.permute(0,1,3,2).flip(2) # 270 degree
            if dgr not in ["0", "90", "180", "270"]:
                print('ERROR degree!')
                exit()
            # ~~~~~ rotation ~~~~~
            # compute output
            if use_gpu: # inputs shape: torch.Size([256, 3, 84, 84])
                inputs = inputs.cuda() 
                labels = labels.cuda()
            outputs,_ = model(inputs)
            outputs = outputs.cpu().data.numpy()
            
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)
    
        all_info = output_dict
        save_pickle(feat_file_name, all_info) # change feature file name here!
        return all_info

if __name__ == '__main__':
    params             = parse_args('test')
    ## params.model    = 'WideResNet28_10' # 'Conv4S' 'WideResNet28_10'
    ## params.method   = 'S2M2_R' # 'S2M2_R' 'rotation' 'manifold_mixup'

    loadfile_base   = configs.data_dir[params.dataset] + 'base.json'
    loadfile_novel  = configs.data_dir[params.dataset] + 'novel.json'
    loadfile_val    = configs.data_dir[params.dataset] + 'val.json'

    img_size    = params.re_size
    TF          = {'True':True, 'False':False}
    flip        = TF[params.flip] 
    
    if params.dataset == 'miniImagenet' or params.dataset == 'CUB' or params.dataset == 'tieredImageNet':
        datamgr       = SimpleDataManager(img_size, batch_size = 100) # 80 is s2m2's and 84 is DC's default
    elif params.dataset == 'cifar':
        datamgr       = SimpleDataManager(img_size, batch_size = 256) # 32 is default
    elif params.dataset == 'MultiDigitMNIST':
        datamgr       = SimpleDataManager(img_size, batch_size = 256) # 64 is default
        
    base_loader   = datamgr.get_data_loader(loadfile_base, aug=False, shf=False)
    novel_loader  = datamgr.get_data_loader(loadfile_novel, aug = False, shf=False)
    val_loader    = datamgr.get_data_loader(loadfile_val, aug = False, shf=False)

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir,
                                                 params.dataset,
                                                 params.model,
                                                 params.method)
    print('checkpoint_dir', checkpoint_dir)
    modelfile   = get_resume_file(checkpoint_dir, fetch_epoch = params.fetch_epoch)

    if params.model == 'WideResNet28_10':
        if params.dataset == 'cifar':
            model = wrn_mixup_model.wrn28_10(num_classes=64, 
                                       loss_type='softmax' if (params.method=='S2M2_R') else 'dist')
        else:
            model = wrn_mixup_model.wrn28_10(num_classes=params.num_classes)
    elif params.model == 'Conv4S':
        model = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'softmax') # softmax | dist

    if use_gpu:
        model = model.cuda()
    cudnn.benchmark = True

    print('> loading model from:', modelfile)
    checkpoint    = torch.load(modelfile)
    state         = checkpoint['state']
    state_keys    = list(state.keys())
    print('----------\n', state_keys, '\n----------\n')     

    callwrap = False
    if 'module' in state_keys[0]:
        callwrap = True
    if callwrap:
        model = WrappedModel(model)
    
    this_model_dict = model.state_dict()
    this_model_dict.update(state)
    model_dict_     = copy.deepcopy(this_model_dict)
    for key in this_model_dict:
        if 'linear.' in key:
            model_dict_[key.replace('linear.','classifier.')] = model_dict_.pop(key)    
    if params.method in ['rotation'] and params.dataset != 'MultiDigitMNIST':
        for key in this_model_dict:
            if 'feature.' in key:
                model_dict_[key.replace('feature.','')] = model_dict_.pop(key)
        print(list(model_dict_.keys()))
    model.load_state_dict(model_dict_) # not this_model_dict
    model.eval()
    if 'b' in params.bvn:
        output_dict_base = extract_feature(base_loader, model, checkpoint_dir, 
                                        params.dataset, dgr=params.dgr,
                                        tag='last', set='base', redo=True, flip=flip,
                                        img_size=img_size)
        print("base set features saved!")
    
    if 'v' in params.bvn:
        output_dict_val = extract_feature(val_loader, model, checkpoint_dir, 
                                        params.dataset, dgr=params.dgr,
                                        tag='last',set='val', redo=True, flip=flip,
                                        img_size=img_size)
        print("val features saved!")
    
    if 'n' in params.bvn:
        output_dict_novel = extract_feature(novel_loader, model, checkpoint_dir, 
                                        params.dataset, dgr=params.dgr,
                                        tag='last',set='novel', redo=True, flip=flip,
                                        img_size=img_size)
        print("novel features saved!")
