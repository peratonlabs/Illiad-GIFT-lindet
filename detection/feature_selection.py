import os

import pickle
import torch
import numpy as np
# import lightgbm as lgb
import os
import pickle
from utils import utils
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
# import pandas

import torchvision
from torchvision.models import resnet50, mobilenet_v2
import timm
from timm.models.vision_transformer import VisionTransformer
import model_factories

def get_arch(model_filepath):
    if torch.cuda.is_available():
        model = torch.load(model_filepath)
    else:
        model = torch.load(model_filepath, map_location=torch.device('cpu'))

    if isinstance(model, model_factories.NerLinearModel):
        numparams = len([p for p in model.parameters()])
        cls = [p for p in model.modules()][1].__class__.__name__ + '_' + str(numparams)
    elif isinstance(model, model_factories.LstmLinearModel) or isinstance(model, model_factories.GruLinearModel) or isinstance(model, model_factories.FCLinearModel):
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
    else:
        numparams = len(model['state_dict'])
        cls = model['model'] + '_' + str(numparams)
    # print(cls)
    return cls


def get_archmap(model_filepaths, metadata_path):

    if metadata_path is not None:
        import pandas
        m = pandas.read_csv(metadata_path)
        ids = m['model_name'].tolist()
        archs = m['model_architecture'].tolist()

        arch_list = []
        for model_filepath in model_filepaths:
            this_id = os.path.split(os.path.split(model_filepath)[0])[1]
            index = ids.index(this_id)
            arch_list.append(archs[index])
    else:
        arch_list = [get_arch(model_filepath) for model_filepath in model_filepaths] 
    arch_map = {}
    for i, arch in enumerate(arch_list):
        if arch not in arch_map:
            arch_map[arch] = []
        # arch_map[arch].append([model_filepaths[i], cls[i]])
        arch_map[arch].append(i)
    print("found the following architectures:", arch_map.keys())
    return arch_map



def arch_train(arch_fns, arch_classes, cfg_dict, ref_model=None):
    # nfeats = cfg_dict['nfeats']
    cls_type = cfg_dict['cls_type']
    # param_batch_sz = cfg_dict['param_batch_sz']
    # print('starting feat selection')
    # weight_mapping = select_feats(arch_fns, arch_classes, cfg_dict, ref_model=ref_model)
    weight_mapping = select_feats2(arch_fns, arch_classes, cfg_dict, ref_model=ref_model)

    torch.cuda.empty_cache()
    # print('finished feat selection')
    # x = [get_mapped_weights(fn, weight_mapping) for fn in arch_fns]
    x = [get_mapped_weights(fn, weight_mapping, cfg_dict, ref_model=ref_model) for fn in arch_fns]
    torch.cuda.empty_cache()
    x = np.stack(x)
    # print(x.shape)

    # ux =x.mean(axis=0)
    # stdx =x.std(axis=0)
    # x = (x-ux)/stdx #Removed old normalization
    # xstats = [ux, stdx]

    if cls_type == 'LogisticRegression':
        classifier = LogisticRegression(max_iter=1000, C=cfg_dict['C'])
    elif cls_type == 'RandomForestClassifier':
        classifier = RandomForestClassifier(n_estimators=10000)
        # classifier = lgb.LGBMClassifier(boosting_type='rf', n_estimators=10000)

    elif cls_type == 'GB':
        # classifier = lgb.LGBMClassifier(boosting_type='goss', n_estimators=500,max_depth=1)
        # classifier = lgb.LGBMClassifier(boosting_type='goss', n_estimators=500,max_depth=2)
        # classifier = lgb.LGBMClassifier(n_estimators=500,max_depth=2)
        classifier = lgb.LGBMClassifier(n_estimators=1000,max_depth=2)


    elif cls_type == 'calavg':
        classifier = CalAvgCls()
        # classifier = CalAvgCls2()

    else:
        assert False, 'bad classifier type'
    # print(x.shape,arch_classes)
    classifier.fit(x, arch_classes)

    return weight_mapping, classifier


def get_mets(y,x,thr):
    pred = x >= thr
    tp = torch.logical_and(pred == 1, y == 1).sum(axis=0)
    fp = torch.logical_and(pred == 1, y == 0).sum(axis=0)
    tn = torch.logical_and(pred == 0, y == 0).sum(axis=0)
    fn = torch.logical_and(pred == 0, y == 1).sum(axis=0)

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return [fpr, tpr]


def get_auc(x,y):
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    # y = np.tile(y,x.shape[1])
    y = y.reshape(-1,1)
    # t = time.time()
    #print("sorting", x.shape)
    xsorted = x.clone()
    xsorted = xsorted.sort(axis=0)[0]
    #print("done")
    # print('sorting', time.time()- t)
    # if torch.cuda.is_available():
    #     x = x.cuda()
    #     y = y.cuda()
    # xsorted = torch.tensor(xsorted).cuda()


    tprs = []
    fprs = []
    # t = time.time()
    for i in range(x.shape[0]):
        # fpr, tpr = get_mets(y, x, xsorted[i:i+1])
        fpr, tpr = get_mets(y, x, xsorted[i:i+1])
        # print(xsorted[i:i+1])
        tprs.append(tpr.cpu().detach().numpy())
        fprs.append(fpr.cpu().detach().numpy())
    # print('get_mets', time.time()- t)

    tprs = np.stack(tprs)
    fprs = np.stack(fprs)

    tprs = np.stack([tprs[:-1] , tprs[1:]],axis=1)
    fprs = np.stack([fprs[:-1] , fprs[1:]],axis=1)

    # t = time.time()
    auc = 1- (np.sum(np.trapz(tprs, fprs, axis=1), axis=0 )+ 1)
    # auc = np.sum(np.trapz(tprs, fprs, axis=1), axis=0 )+ 1 - 0.5
    # print('trap',time.time() - t)

    return auc


def get_corr(x, y):

    x = x.cuda()
    y = y.cuda()

    y = y.reshape(-1,1).float()
    ux = x.mean(axis=0).reshape(1,-1)
    uy = y.mean(axis=0).reshape(1,-1)

    stdx = x.std(axis=0).reshape(1,-1)+1e-8   #* (y.shape[0])/(y.shape[0]-1)
    stdy = y.std(axis=0).reshape(1,-1)+1e-8   #* (y.shape[0])/(y.shape[0]-1)

    cov = (x-ux) * (y-uy)
    cov = cov.sum(axis=0)/(y.shape[0]-1)

    corr = cov/(stdx*stdy* (y.shape[0])/(y.shape[0]-1))

    corr = corr.cpu().detach().numpy().reshape(-1)

    return corr


def get_metric_batched(x,y, maxelements=600000000, fun=get_corr):
    maxfeats = round(maxelements / x.shape[0])

    bi = 0
    this_param_metric = []
    while True:

        if bi > x.shape[1]:
            break

        this_batch_metric = fun(x[:, bi:bi + maxfeats], y)

        this_param_metric.append(this_batch_metric)
        bi += maxfeats

    return np.concatenate(this_param_metric)


def get_deltas(model_or_path, ref_model=None, norm=None, psort=False):
    if not isinstance(model_or_path, torch.nn.Module):
        if torch.cuda.is_available():
            model = torch.load(model_or_path)
        else:
            model = torch.load(model_or_path, map_location=torch.device('cpu'))
    else:
        model = model_or_path
    if torch.cuda.is_available():
        model.cuda()

    if ref_model is None:
        ps = [mp.data for mp in model.parameters()]
    else:
        ps = []
        for p, ref_p in zip(model.parameters(), ref_model.parameters()):
            if p.shape == ref_p.shape:
                ps.append(p.data-ref_p.data)

    if norm is not None:
        if norm=='pnorm':
            ps_new = []
            for p in ps:
                std = p.std()
                if std==0 or std.isnan().any():
                    ps_new.append(p)
                else:
                    ps_new.append(p/std)
            ps = ps_new
        else:
            if norm == 'white':
                scaling_factor = torch.std(torch.cat([p.reshape(-1) for p in ps]))
            elif norm == 'cosine':
                vec = torch.cat([p.reshape(-1) for p in ps])
                scaling_factor = torch.norm(vec)/np.sqrt(vec.shape[0])
            else:
                scaling_factor = norm

            ps = [p/scaling_factor for p in ps]
    
    if psort:
        ps = [p.sort()[0] for p in ps]

    return ps


def proc_feat_type(feat_type, ref_model):
    if feat_type == 'raw':
        norm_type = None
    elif feat_type == 'cosine_delta' or feat_type == 'cosine':
        norm_type = 'cosine'
    elif feat_type == 'white_delta' or feat_type == 'white':
        norm_type = 'white'
    elif feat_type == 'pnorm_delta' or feat_type == 'pnorm':
        norm_type = 'pnorm'

    if "_delta" not in feat_type:
        ref_model = None

    return norm_type, ref_model


def get_mapped_weights(model_filepath, weight_mapping, cfg_dict, ref_model=None):
    # print(ref_model)

    norm_type, ref_model = proc_feat_type(cfg_dict['features'], ref_model)

    if cfg_dict['normalize_for_feature_selection']:
        post_norm = None
        feat_sel_norm = norm_type
    else:
        post_norm = norm_type
        feat_sel_norm = None
    
    with torch.no_grad():
        ps = get_deltas(model_filepath, ref_model=ref_model, norm=feat_sel_norm, psort=cfg_dict["sort_tensors"])
    
    mapped_weights = []

    for i in range(len(weight_mapping)):
        param = (ps[i]).cpu().detach().numpy()
        param = param.reshape(-1)
        mapped_weights.append(param[weight_mapping[i]])

    mapped_weights = np.concatenate(mapped_weights)
    if post_norm == 'cosine':
        mapped_weights = mapped_weights * np.sqrt(mapped_weights.shape[0]) / np.linalg.norm(mapped_weights)
    elif post_norm == 'white':
        mapped_weights = mapped_weights / mapped_weights.std()
    else:
        assert post_norm is None, "bad normalization option"

    return mapped_weights


def get_params(models, start_ind, num_params, ref_model=None, norm=None, pinds=None, psort=False):

    output_ps = []
    for model in models:
        ps = get_deltas(model, ref_model=ref_model, norm=norm, psort=False)
        if pinds is None:
            ps = ps[start_ind:start_ind+num_params]
        else:
            ps = [ps[pind] for pind in pinds]
        # ps = [p.cpu().detach().numpy().reshape(-1) for p in ps]
        ps = [p.reshape(-1).cpu() for p in ps]
        

        if len(output_ps)==0:
            output_ps = [[p] for p in ps]
        else:
            for i, p in enumerate(ps):
                if output_ps[i][0].shape[0] == p.shape[0]:
                    output_ps[i].append(p)
                else:
                    num_params = i
                    output_ps = output_ps[:i]
                    break

    # output_ps = [np.stack(vectors) for vectors in output_ps]
    # if flatten:
    output_ps = [torch.stack(vectors) for vectors in output_ps]
    # else:
    #     output_ps = [vectors for vectors in output_ps]


    return output_ps


def detect(fn, weight_mapping, classifier, cfg_dict, ref_model=None):
    # x = [get_mapped_weights(fn, weight_mapping)]
    # ux, stdx = xstats
    # x = (x - ux) / stdx
    x = [get_mapped_weights(fn, weight_mapping, cfg_dict, ref_model=ref_model)]
    p = classifier.predict_proba(x)[:, 1]
    return p

def cv_arch_train(arch_fns, arch_classes, cfg_dict, holdout_ratio=0.1, num_cvs=10, cv_scratch_dir=None, arch=None, ref_model=None):

    # x1,x2,y1,y2 = get_good_split(arch_fns, arch_classes, holdout_ratio=holdout_ratio, ignore_val=True)
    ns = arch_classes.shape[0]
    inds = np.arange(ns)
    # split_ind = round((1 - holdout_ratio) * ns)
    # print(ns, split_ind)
    # if split_ind == ns:
    #     split_ind = ns-1

    cvcal_scores = []
    truths = []
    for i in range(num_cvs):
        tr_fns,v_fns,tr_cls,v_cls = get_good_split(arch_fns, arch_classes, holdout_ratio=holdout_ratio, ignore_val=True)



        # np.random.shuffle(inds)
        # trinds = inds[:split_ind]
        # vinds = inds[split_ind:]

        # tr_fns = arch_fns[trinds]
        # tr_cls = arch_classes[trinds]
        # v_fns = arch_fns[vinds]
        # v_cls = arch_classes[vinds]

        # dump_fn = None if cv_scratch_dir is  None else os.path.join(cv_scratch_dir, 'cvdump' + str(i) + '.p')

        weight_mapping, classifier = arch_train(tr_fns, tr_cls, cfg_dict, ref_model=ref_model)
        if cv_scratch_dir is not None:
            os.makedirs(cv_scratch_dir, exist_ok=True)

            # arch_name = arch.split('.')[-1][:-2]
            arch_name = arch


            dump_fn = os.path.join(cv_scratch_dir, 'cvdump' + '_' + arch_name + '_' + str(i) + '.p')

            xtr = [get_mapped_weights(fn, weight_mapping, cfg_dict, ref_model=ref_model) for fn in tr_fns]
            xtr = np.stack(xtr)
            xv = [get_mapped_weights(fn, weight_mapping, cfg_dict, ref_model=ref_model) for fn in v_fns]
            if len(xv)>0:
                xv = np.stack(xv)

            # dump_data = [xtr, tr_cls, xv, v_cls]
            xstats = None

            with open(dump_fn, 'wb') as f:
                pickle.dump([weight_mapping, classifier, xstats, tr_fns, v_fns, xtr, tr_cls, xv, v_cls], f)

        # xv = [get_mapped_weights(fn, weight_mapping) for fn in v_fns]

        pv = [detect(fn, weight_mapping, classifier, cfg_dict, ref_model=ref_model) for fn in v_fns]

        # pv = classifier.predict_proba(xv)[:, 1]
        try:
            print(roc_auc_score(v_cls, pv), log_loss(v_cls, pv))
        except:
            print('AUC error (probably due to class balance)')
        cvcal_scores.append(pv)
        truths.append(v_cls)

    torch.cuda.empty_cache()
    weight_mapping, classifier = arch_train(arch_fns, arch_classes, cfg_dict, ref_model=ref_model)
    # arch_weight_mappings[arch] = weight_mapping
    # arch_classifiers[arch] = classifier
    # arch_xstats[arch] = xstats
    return weight_mapping, classifier, cvcal_scores, truths


def cv_train(model_filepaths, cls, cfg_dict, num_cvs=10, holdout_ratio=0.1, cv_scratch_dir=None, metadata_path=None, ref_model_map=None):

    arch_map = get_archmap(model_filepaths, metadata_path)
    #print(arch_map.keys())
    arch_weight_mappings = {}
    arch_classifiers = {}
    # arch_xstats = {}

    for arch, arch_inds in arch_map.items():
        print('starting arch', arch)
        arch_fns = np.array([model_filepaths[i] for i in arch_inds])
        arch_classes = np.array([cls[i] for i in arch_inds])
        if ref_model_map:
            ref_model = ref_model_map[arch]
        else:
            ref_model = None

        weight_mapping, classifier, pvs, yvs = cv_arch_train(arch_fns, arch_classes, cfg_dict, holdout_ratio=holdout_ratio,
                                                   num_cvs=num_cvs, cv_scratch_dir=cv_scratch_dir, arch=arch,
                                                   ref_model=ref_model)
        arch_weight_mappings[arch] = weight_mapping
        arch_classifiers[arch] = classifier

    return arch_weight_mappings, arch_classifiers


def select_feats(model_fns, labels, cfg_dict, ref_model=None):
    labels = torch.tensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()
    nfeats = cfg_dict['nfeats']
    criterion = cfg_dict['feature_selection_criterion']
    param_batch_sz = cfg_dict['param_batch_sz']

    norm_type, ref_model = proc_feat_type(cfg_dict['features'], ref_model)
    if cfg_dict['normalize_for_feature_selection']:
        norm = norm_type
    else:
        norm = None

    ind = 0
    aucs = []
    while True:
        # print('starting param', ind)
        xs = get_params(model_fns, ind, param_batch_sz, ref_model=ref_model, norm=norm)
        torch.cuda.empty_cache()
        xs_len = len(xs)

        # print('got params')
        # print(sum([np.product(xx.shape) for xx in xs]))

        iii=0
        for x in xs:
            # print('computing aucs',iii)
            iii+=1
            if criterion == 'auc':
                this_aucs = np.abs(get_metric_batched(x, labels, fun=get_auc, maxelements=200000000).astype(np.float64) - 0.5)
            elif criterion == 'corr':
                this_aucs = np.abs(get_metric_batched(x, labels, fun=get_corr, maxelements=200000000).astype(np.float64))
                # this_aucs = np.abs(get_auc(x, labels).astype(np.float64) - 0.5)
            else:
                assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
            this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)
            aucs.append(this_aucs)
            torch.cuda.empty_cache()

        del xs
        # import gc
        # gc.collect()

        if xs_len < param_batch_sz:
            break

        ind += param_batch_sz

    aucscopy = np.concatenate(aucs)
    aucscopy.sort()
    if aucscopy.shape[0]>nfeats:
        thr = aucscopy[-nfeats]
    else:
        thr = -np.inf

    # featmap = aucs >= thr

    weight_mapping = []
    for auc in aucs:
        weight_mapping.append(auc >= thr)

    torch.cuda.empty_cache()
    return weight_mapping
    # x = [get_mapped_weights(fn, weight_mapping) for fn in arch_fns]
    #
    # x = np.stack(x)


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


def select_pinds(model_fns, labels, cfg_dict, norm, ref_model=None):

    # labels = torch.tensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()
    nfeats = cfg_dict['nfeats']
    criterion = cfg_dict['feature_selection_criterion']
    param_batch_sz = cfg_dict['param_batch_sz']
    

    ntensors = cfg_dict['ntensors']
    # ntrials = cfg_dict['ntrials']
    # holdout_ratio = cfg_dict['holdout_ratio']

    # norm = 'pnorm'
    ntrials = 10
    holdout_ratio = 0.2
    # ntensors = 25
    # if ntensors == 0:
    #     return select_feats(model_fns, labels, cfg_dict, ref_model=ref_model)

    nfeats = round(nfeats/ntensors)

    ind = 0

    param_aucs = []

    while True:
        xs = get_params(model_fns, ind, param_batch_sz, ref_model=ref_model, norm=norm)
        torch.cuda.empty_cache()
        xs_len = len(xs)
        
        for x in xs:
            this_param_aucs = []
            for trial in range(ntrials):
                x_tr, x_val, labels_tr, labels_val = get_good_split(x, labels, holdout_ratio=holdout_ratio)


                if criterion == 'auc':
                    this_aucs = np.abs(get_metric_batched(x_tr, labels_tr, fun=get_auc, maxelements=200000000).astype(np.float64) - 0.5)
                elif criterion == 'corr':
                    this_aucs = np.abs(get_metric_batched(x_tr, labels_tr, fun=get_corr, maxelements=200000000).astype(np.float64))
                else:
                    assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
                this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)

                aucscopy = this_aucs.copy()
                aucscopy.sort()
                if aucscopy.shape[0]>nfeats:
                    thr = aucscopy[-nfeats]
                else:
                    thr = -np.inf
                cur_weight_mapping = this_aucs >= thr

                feats_tr = x_tr[:,cur_weight_mapping].detach().cpu().numpy()
                feats_val = x_val[:,cur_weight_mapping].detach().cpu().numpy()

                labels_tr = labels_tr.detach().cpu().numpy()
                labels_val = labels_val.detach().cpu().numpy()

                classifier = LogisticRegression(max_iter=1000, C=0.3)
                classifier.fit(feats_tr, labels_tr)
                pv = classifier.predict_proba(feats_val)[:, 1]

                this_param_auc = roc_auc_score(labels_val, pv)
                this_param_aucs.append(this_param_auc)
            param_aucs.append(np.mean(this_param_aucs))

        if xs_len < param_batch_sz:
            break
        ind += param_batch_sz
    pinds = np.argsort(param_aucs)

    pinds = pinds[-ntensors:]
    return pinds


def select_feats2(model_fns, labels, cfg_dict, ref_model=None):

    if cfg_dict['ntensors'] == 0:   
        return select_feats(model_fns, labels, cfg_dict, ref_model=ref_model)
    
    labels = torch.tensor(labels)
    if torch.cuda.is_available():
        labels = labels.cuda()
    nfeats = cfg_dict['nfeats']
    criterion = cfg_dict['feature_selection_criterion']
    param_batch_sz = cfg_dict['param_batch_sz']

    norm_type, ref_model = proc_feat_type(cfg_dict['features'], ref_model)
    if cfg_dict['normalize_for_feature_selection']:
        norm = norm_type
    else:
        norm = None

    # nfeats = 100

    pinds = select_pinds(model_fns, labels, cfg_dict, norm, ref_model=ref_model)
    weight_mapping = [[] for i in range(1+max(pinds))]

    xs = get_params(model_fns, 0, param_batch_sz, ref_model=ref_model, norm=norm, pinds=pinds)

    torch.cuda.empty_cache()
    xs_len = len(xs)

    for ii, x in enumerate(xs):
        pind = pinds[ii]

        if criterion == 'auc':
            this_aucs = np.abs(get_metric_batched(x, labels, fun=get_auc, maxelements=200000000).astype(np.float64) - 0.5)
        elif criterion == 'corr':
            this_aucs = np.abs(get_metric_batched(x, labels, fun=get_corr, maxelements=200000000).astype(np.float64))
        else:
            assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
        this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)
        aucscopy = this_aucs.copy()
        aucscopy.sort()
        if aucscopy.shape[0]>nfeats:
            thr = aucscopy[-nfeats]
        else:
            thr = -np.inf
        cur_weight_mapping = this_aucs >= thr

        weight_mapping[pind] = cur_weight_mapping
        torch.cuda.empty_cache()

    del xs

    torch.cuda.empty_cache()
    return weight_mapping


class CalAvgCls():

    def fit(self, x, y):
        self.ux = x.mean(axis=0)
        self.stdx = x.std(axis=0)
        corr = get_corr(x, y)
        self.sign = np.sign(corr)

    def predict_proba(self, x):
        p = (self.sign * (x - self.ux) / self.stdx).mean(axis=1)
        p = 1/(1 + np.exp(-p))
        return np.stack([1-p,p], axis=1)


class CalAvgCls2():

    def fit(self, x, y):
        self.ux = x.mean(axis=0)
        self.stdx = x.std(axis=0)
        corr = get_corr(x, y)
        self.sign = np.sign(corr)
        self.y = y
        self.ref_feats = (self.sign * (x - self.ux) / self.stdx)
        # self.ref_feats.sort()
        # self.probs = np.arange(len(self.ref_feats)) / float(len(self.ref_feats))
        # self.cdfs = [ECDF(self.ref_feats[:,i]) for i in range(self.ref_feats.shape[1])]


    def predict_proba(self, x):
        feats = (self.sign * (x - self.ux) / self.stdx)

        # y = np.arange(len(x)) / float(len(x))

        # self.ref_feats

        # p = [f, ref in ]
        calfeats = 0.0*feats + 1.0

        for row in range(feats.shape[0]):
            for col in range(feats.shape[1]):
                # inds = np.where(self.ref_feats>feats[row, col])[1]
                ps = self.y[self.ref_feats[:,col]>feats[row, col]]
                if ps.shape[0]>0:
                    calfeats[row,col] = ps.mean()

        p = calfeats.mean(axis=1)
        # pp=1
        # p = (self.sign * (x - self.ux) / self.stdx).mean(axis=1)
        # p = 1/(1 + np.exp(-p))
        return np.stack([1-p,p], axis=1)

