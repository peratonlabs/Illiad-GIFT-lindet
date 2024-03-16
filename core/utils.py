
import torch
import torchvision
import collections
import model_factories
import os
import numpy as np

def get_arch(model_or_path):
    if type(model_or_path) == str:
        if torch.cuda.is_available():
            model = torch.load(model_or_path)
        else:
            model = torch.load(model_or_path, map_location=torch.device('cpu'))
    else:
        model = model_or_path

    if isinstance(model, model_factories.NerLinearModel):
        numparams = len([p for p in model.parameters()])
        cls = [p for p in model.modules()][1].__class__.__name__ + '_' + str(numparams)
    elif isinstance(model, model_factories.LstmLinearModel) or isinstance(model, model_factories.GruLinearModel) or isinstance \
            (model, model_factories.FCLinearModel):
        params = [p for p in model.parameters()]
        numparams = len(params)
        szparams = params[1].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(szparams)

    elif isinstance(model, torchvision.models.resnet.ResNet):
        params = [p for p in model.parameters()]
        numparams = len(params)
        width = params[4].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(width)

    elif isinstance(model, torchvision.models.shufflenetv2.ShuffleNetV2):
        params = [p for p in model.parameters()]
        numparams = len(params)
        width = params[7].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(width)

    elif isinstance(model, torchvision.models.squeezenet.SqueezeNet):
        params = [p for p in model.parameters()]
        numparams = len(params)
        width = params[1].shape[0]
        cls = model.__class__.__name__ + '_' + str(numparams) + '_' + str(width)

    elif isinstance(model, torch.nn.Module):
        numparams = len([p for p in model.parameters()])
        cls = model.__class__.__name__ + '_' + str(numparams)
    elif isinstance(model, collections.OrderedDict):
        numparams = len(model)
        cls = model.__class__.__name__ + '_' + str(numparams)
    else:
        numparams = len(model['state_dict'])
        cls = model['model'] + '_' + str(numparams)
    # print(cls)
    return cls


def get_archmap(model_filepaths):
    # finds archs for all model_filepaths and put into a dictionary mapping arch -> paths
    arch_list = [get_arch(model_filepath) for model_filepath in model_filepaths]
    arch_map = {}
    for i, arch in enumerate(arch_list):
        if arch not in arch_map:
            arch_map[arch] = []
        arch_map[arch].append(i)
    print("found the following architectures:", arch_map.keys())
    return arch_map


def proc_feat_type(feat_type, ref_model):
    if feat_type == 'raw':
        norm_type = None
    elif feat_type == 'cosine_delta' or feat_type == 'cosine':
        norm_type = 'cosine'
    elif feat_type == 'white_delta' or feat_type == 'white':
        norm_type = 'white'
    elif feat_type == 'pnorm_delta' or feat_type == 'pnorm':
        norm_type = 'pnorm'
    else:
        assert False, f"bad feat_type {feat_type}"

    if "_delta" not in feat_type:
        ref_model = None

    return norm_type, ref_model


def get_class_allrounds(modeldir):
    # 1 -> poisoned, 0 -> clean
    truthfile = os.path.join(modeldir,'ground_truth.csv')

    assert os.path.exists(truthfile), truthfile + " doesn't exist"
    with open(truthfile) as f:
        truth = int(f.read())
    return truth


def get_good_split(x, labels, holdout_ratio=0.1, ignore_val=False):

    # print((labels==0).sum(), (labels==1).sum())

    if ignore_val:
        assert (labels==0).sum()>0 and (labels==1).sum()>0, "can't split data without multiple examples of each class"
    else:
        assert (labels==0).sum()>1 and (labels==1).sum()>1, "can't split data without multiple examples of each class"

    ns = x.shape[0]
    inds = np.arange(ns)
    split_ind = round((1 - holdout_ratio) * ns)

    if not ignore_val and split_ind > ns-2:
        split_ind = ns-2

    labels_tr = []
    labels_val = []

    # print(ns, split_ind)
    # print(split_ind)



    while 0 not in labels_tr or 1 not in labels_tr or ( not ignore_val and (0 not in labels_val or 1 not in labels_val)):
        # print((0 not in labels_tr or 1 not in labels_tr) )
        np.random.shuffle(inds)
        trinds = inds[:split_ind]
        vinds = inds[split_ind:]

        x_tr = x[trinds]
        x_val = x[vinds]
        labels_tr = labels[trinds]
        labels_val = labels[vinds]
    return x_tr, x_val, labels_tr, labels_val