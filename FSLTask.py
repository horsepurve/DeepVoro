import os
import pickle
import numpy as np
import torch
# from tqdm import tqdm

# ========================================================
#   Usefull paths
''' S2M2_R '''
_datasetFeaturesFilesS = {
    "miniImagenet"    : "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last/novel_features.plk",
    "CUB"             : "./checkpoints/CUB/WideResNet28_10_S2M2_R/last/novel_features.plk",
    "cifar"           : "./checkpoints/cifar/WideResNet28_10_S2M2_R/last/novel_features.plk",
    "MultiDigitMNIST" : "./checkpoints/MultiDigitMNIST/WideResNet28_10_S2M2_R/last/novel_features.plk",
    "tieredImageNet"  : "./checkpoints/tieredImageNet/WideResNet28_10_S2M2_R/last/novel_features.plk"
    }


''' rotation '''
_datasetFeaturesFilesR = {
    "miniImagenet"    : "./checkpoints/miniImagenet/WideResNet28_10_rotation/last/novel_features.plk",
    "CUB"             : "./checkpoints/CUB/WideResNet28_10_rotation/last/novel_features.plk",
    "cifar"           : "./checkpoints/cifar/WideResNet28_10_rotation/last/novel_features.plk",
    "MultiDigitMNIST" : "./checkpoints/MultiDigitMNIST/Conv4S_rotation/last/novel_features.plk",
    "tieredImageNet"  : "./checkpoints/tieredImageNet/WideResNet28_10_rotation/last/novel_features.plk"
    }


''' manifold_mixup '''
_datasetFeaturesFilesM = {
    "miniImagenet"    : "./checkpoints/miniImagenet/WideResNet28_10_manifold_mixup/last/novel_features.plk",
    "CUB"             : "./checkpoints/CUB/WideResNet28_10_manifold_mixup/last/novel_features.plk",
    "cifar"           : "./checkpoints/cifar/WideResNet28_10_manifold_mixup/last/novel_features.plk",
    "MultiDigitMNIST" : "./checkpoints/MultiDigitMNIST/WideResNet28_10_manifold_mixup/last/novel_features.plk",
    "tieredImageNet"  : "./checkpoints/tieredImageNet/WideResNet28_10_manifold_mixup/last/novel_features.plk"
    }

''' Cross '''
_datasetFeaturesFilesC = {
    "miniImagenet"    : "./checkpoints/CUB/WideResNet28_10_S2M2_R/cross/novel_features.plk",
    "CUB"             : "./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/cross/novel_features.plk",
    }

_cacheDir = "./cache"
_maxRuns = 10000
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None


def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(dsname,
                dgr="",
                mth='s2m2',
                flip=False,
                img_size=84,
                nv='novel',
                backbone=''):
    '''
    suffix is for rotation, etc.

    dsname="miniImagenet" # "MultiDigitMNIST"
    suffix=""
    mth='s2m2'
    '''

    if mth == 's2m2':
        _datasetFeaturesFiles = _datasetFeaturesFilesS
    elif mth == 'rot':
        _datasetFeaturesFiles = _datasetFeaturesFilesR
    elif mth == 'mix':
        _datasetFeaturesFiles = _datasetFeaturesFilesM
    elif mth == 'mae':
        _datasetFeaturesFiles = _datasetFeaturesFilesMA
    elif mth == 'simple':
        _datasetFeaturesFiles = _datasetFeaturesFilesSS
    elif mth == 'cross':
        _datasetFeaturesFiles = _datasetFeaturesFilesC
    else:
        print('ERROR method!!')
        exit(0)
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labs, labels, _randStates, _rsCfg, _min_examples
    global _max_examples, _n_examples, _n_examples_l, _dim, _n_l

    dsName      = dsname
    _randStates = None
    _rsCfg      = None
    _n_l        = 0

    # Loading data from files on computer
    # home = expanduser("~")
    feat_file_name = '{}_s{}_r{}_f{}.plk'.format(_datasetFeaturesFiles[dsname][:-4],
                                                 img_size,
                                                 dgr, # former: '0' if not dgr else dgr,
                                                 '1' if flip else '0')
    if dgr == 'c10': # only for MultiDigitMNIST
        feat_file_name = _datasetFeaturesFiles[dsname][:-4]+'.plk'+dgr
    if nv == 'val':
        feat_file_name = feat_file_name.replace('novel', 'val')
    if backbone != '':
        feat_file_name = feat_file_name.replace('BONE', backbone)
    print('>>> reading', feat_file_name)
    extracted_feature = feat_file_name
    dataset           = _load_pickle(extracted_feature)
    _dim              = dataset["data"].shape[1]
    print('all labels:', dataset['labels'].numpy())
    #%% Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    _max_examples = 0
    _n_examples   = dict()
    '''
    for i in range(dataset["labels"].shape[0]):
        _l_this = dataset["labels"][i].tolist()
        _l_len  = torch.where(dataset["labels"] == _l_this)[0].shape[0]
        if _l_len > 0:
            _min_examples = min(_min_examples, _l_len)
            _max_examples = max(_max_examples, _l_len)
            if _l_this not in _n_examples:
                _n_examples[_l_this] = _l_len
    '''
    for la in set(dataset["labels"].numpy()):
        _l_len  = torch.where(dataset["labels"] == la)[0].shape[0]
        if _l_len > 0:
            _min_examples = min(_min_examples, _l_len)
            _max_examples = max(_max_examples, _l_len)
            if la not in _n_examples:
                _n_examples[la] = _l_len
    _n_l = len(_n_examples)
    print("min/max numbers per class over {} classes: {:d}/{:d}\n".format(_n_l,
                                                                          _min_examples,
                                                                          _max_examples))

    #%% Generating data tensors
    # data = torch.zeros((0, _min_examples, _dim))
    data = torch.zeros((_n_l, _max_examples, _dim))
    labs = []

    labels        = dataset["labels"].clone()
    l_i           = 0
    _n_examples_l = []
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        labs.append(labels[0].numpy())
        # data = torch.cat([data,
        #                   dataset["data"][indices, :][:_min_examples].view(1, _min_examples, -1)], dim=0)
        _this_class = dataset["data"][indices, :]
        data[l_i][:_this_class.shape[0],:] = _this_class # TODO: here idx can be different!
        _n_examples_l.append(_this_class.shape[0])
        l_i += 1
        indices = torch.where(labels != labels[0])[0]
        labels  = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))
    labs = np.array(labs)
    print('labels in order:', labs)
    return feat_file_name
#%%
def GenerateRun(iRun,
                cfg,
                regenRState=False,
                generate=True,
                re_true_lab=False):
    global _randStates, data, _min_examples, labs
    global _max_examples, _n_examples, _n_examples_l, _dim, _n_l

    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes         = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = [np.arange(_n_examples_l[c]) for c in classes]
    dataset         = None
    if generate:
        dataset = torch.zeros((cfg['ways'],
                               cfg['shot']+cfg['queries'],
                               data.shape[2]))
    idx_this = []
    for i in range(cfg['ways']):
        shuffle_indice = np.random.permutation(shuffle_indices[i])
        # print(classes[:5], shuffle_indices[:5])
        if generate:
            dataset[i] = data[classes[i], shuffle_indice,:][:cfg['shot']+cfg['queries']]
            idx_this.append(shuffle_indice[:cfg['shot']+cfg['queries']])
    if re_true_lab:
        return dataset, labs[classes], np.array(idx_this)
    else:
        return dataset
#%%
def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None,
                   end=None,
                   cfg=None,
                   re_true_lab=False):
    global dataset, _maxRuns, labset
    labset = []
    idxset = []
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun], lab_this, idx_this = GenerateRun(start+iRun, cfg, re_true_lab=True)
        labset.append(lab_this)
        idxset.append(idx_this)
    labset = np.array(labset)
    idxset = np.array(idxset)
    if re_true_lab:
        return dataset, labset, idxset
    else:
        return dataset


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())
