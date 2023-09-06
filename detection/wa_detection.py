import os
import pickle
import numpy as np
from utils import utils
from joblib import dump, load
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, log_loss
from detection.feature_selection import detect, get_arch, get_archmap, arch_train, get_mapped_weights, cv_arch_train
import torch
from torchvision.models import resnet50, mobilenet_v2
import timm




def cal(arg_dict, metaParameters, ref_model_function=None):

    base_path = os.path.join(arg_dict['configure_models_dirpath'], 'models')
    modeldirs = os.listdir(base_path)
    # modeldirs.sort()
    model_filepaths = [os.path.join(base_path, modeldir, 'model.pt') for modeldir in modeldirs]
    # cls = [utils.get_class(os.path.join(base_path, modeldir, 'config.json')) for modeldir in modeldirs]
    cls = [utils.get_class_allrounds(os.path.join(base_path, modeldir)) for modeldir in modeldirs]

    # cfg_dict={'nfeats': 1000, 'cls_type': 'LogisticRegression', 'param_batch_sz': 80, 'C': 0.03}
    gift_basepath = arg_dict['gift_basepath']
    holdoutratio=arg_dict['cv_test_prop']

    use_ISO = False
    num_cv_trials = arg_dict['num_cv_trials']

    scratch_dirpath = arg_dict['scratch_dirpath']
    cv_scratch_dir = os.path.join(scratch_dirpath, 'cv_results')
    os.makedirs(cv_scratch_dir, exist_ok=True)

    
    arch_map = get_archmap(model_filepaths, metadata_path=None)
    # cfg_dict_resnet = {'nfeats': 1000, 'cls_type': 'LogisticRegression', 'param_batch_sz': 80, 'C': 100.0}
    # cfg_dict_vit = {'nfeats': 1000, 'cls_type': 'LogisticRegression', 'param_batch_sz': 80, 'C': 0.1}
    # cfg_dict_mobilenet = {'nfeats': 1000, 'cls_type': 'LogisticRegression', 'param_batch_sz': 80, 'C': 100.0}

    cfg_dicts = metaParameters
    # ref_models = {}
    # for arch in arch_map.keys():
    #
    #     ref_models[arch] = ref_model_function(arch)

        # if "ResNet" in arch:
        #     # cfg_dicts[arch] = cfg_dict_resnet
        #     ref_model = resnet50()
        #     ref_model.load_state_dict(torch.load(os.path.join(gift_basepath, 'resnet50_V2.pth')))
        # if "VisionTransformer" in arch:
        #     # cfg_dicts[arch] = cfg_dict_vit
        #     ref_model = timm.create_model('vit_base_patch32_224', pretrained=False)
        #     ref_model.load_state_dict(torch.load(os.path.join(gift_basepath, 'vit.pt')))
        # if "MobileNetV2" in arch:
        #     # cfg_dicts[arch] = cfg_dict_mobilenet
        #     ref_model = mobilenet_v2()
        #     ref_model.load_state_dict(torch.load(os.path.join(gift_basepath, 'mobilenet_V2.pth')))
        # ref_models[arch] = ref_model
    # import pdb; pdb.set_trace()
    arch_weight_mappings = {}
    ISO_arch_classifiers = {}
    arch_classifiers = {}
    # arch_xstats = {}
    print('calibrating for architectures: ', arch_map.keys())

    # import pdb; pdb.set_trace()
    #TODO refactor this code
    # for arch, arch_inds in sorted(arch_map.items(), reverse= False):
    for arch, arch_inds in arch_map.items():
        print('starting arch', arch)

        arch_fns = np.array([model_filepaths[i] for i in arch_inds])
        arch_classes = np.array([cls[i] for i in arch_inds])
        ref_model = ref_model_function(arch)
        if torch.cuda.is_available() and ref_model is not None:
            ref_model.cuda()

        with torch.no_grad():
            weight_mapping, classifier, pvs, yvs = cv_arch_train(arch_fns, arch_classes, cfg_dicts[arch], holdout_ratio=holdoutratio,
                                                       num_cvs=num_cv_trials, cv_scratch_dir=cv_scratch_dir, arch=arch,
                                                       ref_model=ref_model)

        # ns = arch_classes.shape[0]
        # inds = np.arange(ns)
        # split_ind = round((1-holdoutratio)*ns)
        #
        # lr_scores = []
        # truths = []
        # numSample = num_cv_trials
        # for i in range(numSample):
        #     np.random.shuffle(inds)
        #     trinds = inds[:split_ind]
        #     vinds = inds[split_ind:]
        #
        #     tr_fns = arch_fns[trinds]
        #     tr_cls = arch_classes[trinds]
        #     v_fns = arch_fns[vinds]
        #     v_cls = arch_classes[vinds]
        #
        #
        #     weight_mapping, classifier, xstats = arch_train(tr_fns, tr_cls, cfg_dict[arch], gift_basepath, ref_model=ref_model)
        #     if cv_scratch_dir is not None:
        #         os.makedirs(cv_scratch_dir, exist_ok=True)
        #
        #         arch_name = arch.split('.')[-1][:-2]
        #
        #         dump_fn = os.path.join(cv_scratch_dir,'cvdump'+'_'+arch_name+'_'+str(i)+'.p')
        #
        #         xtr = [get_mapped_weights(fn, weight_mapping, gift_basepath) for fn in tr_fns]
        #         xtr = np.stack(xtr)
        #         xv = [get_mapped_weights(fn, weight_mapping, gift_basepath) for fn in v_fns]
        #         xv = np.stack(xv)
        #
        #         # dump_data = [xtr, tr_cls, xv, v_cls]
        #
        #         with open(dump_fn, 'wb') as f:
        #             pickle.dump([weight_mapping, classifier, xstats, tr_fns, v_fns, xtr, tr_cls, xv, v_cls], f)
        #
        #
        #     pv = [detect(fn, weight_mapping, classifier, xstats, gift_basepath) for fn in v_fns]
        #     try:
        #         print(f" AUC: {roc_auc_score(v_cls, pv)}, CE: {log_loss(v_cls, pv)}")
        #     except:
        #         print('AUC error (probably due to class balance)')
        #     lr_scores.append(pv)
        #     truths.append(v_cls)

        if use_ISO:

            ISOce_scores = []

            for _ in range(num_cv_trials):
                ns = len(pvs)
                ind = np.arange(ns)
                np.random.shuffle(ind)
                split = round(  (1-holdoutratio)*ns)

                ptr = np.concatenate(pvs[ind[:split]])
                ptst = np.concatenate(pvs[ind[split:]])
                ytr = np.concatenate(yvs[ind[:split]])
                ytst = np.concatenate(yvs[ind[split:]])

                ir_model = IsotonicRegression(out_of_bounds='clip')
                ir_model.fit(ptr, ytr)
                p2tst = ir_model.transform(ptst)
                p2tst = np.clip(p2tst, 0.01, 0.99)

                ISOce_scores.append(log_loss(ytst, p2tst))

            print('new ISO CE', np.mean(ISOce_scores))


        # weight_mapping, classifier, xstats = arch_train(arch_fns, arch_classes, cfg_dict[arch], gift_basepath)
        arch_weight_mappings[arch] = weight_mapping
        arch_classifiers[arch] = classifier
        # arch_xstats[arch] = None
        if use_ISO:
            ISO_arch_classifiers[arch] = ir_model


    if use_ISO:
        dump([arch_weight_mappings, arch_classifiers, ISO_arch_classifiers, cfg_dicts], os.path.join(arg_dict['learned_parameters_dirpath'], 'wa_lr.joblib'))
    else:
        dump([arch_weight_mappings, arch_classifiers, cfg_dicts], os.path.join(arg_dict['learned_parameters_dirpath'], 'wa_lr.joblib'))



def detector(model_filepath,lr_path, metaParameters, gift_basepath, ref_model_function=None):
    # modelsplit =metaParameters["modelsplit"]
        
    mappings, classifiers, xstats_dict = load(lr_path)
    
    #TODO inferr the model archtecture and select the right weight_mapping from the data. This is a holder code

    arch = get_arch(model_filepath)
    weight_mapping = mappings[arch]
    classifier = classifiers[arch]
    xstats = xstats_dict[arch]
    cfg_dict = metaParameters[arch]
    if ref_model_function is not None:
        ref_model = ref_model_function(arch)

    assert not weight_mapping == mappings,  "Please select an appropriate weight_mapping"
    assert not classifier == classifiers, "Please select an appropriate classifier"
    assert not xstats == xstats_dict, "Please select an appropriate xstats"

    pv = detect(model_filepath, weight_mapping, classifier, cfg_dict, ref_model=ref_model)
        
    return pv


def det(arg_dict, metaParameters, ref_model_function=None, train=False):

    learned_parameters_dirpath = arg_dict["learned_parameters_dirpath"]
    basepath = arg_dict["gift_basepath"]
    detname = "wa_lr.joblib"
    # metaParameters = utils.read_json(arg_dict['metaparameters_filepath'])

    lr_path = os.path.join(learned_parameters_dirpath, detname) 
    lr_path = os.path.join(basepath, lr_path)

    if train:
        cal(arg_dict, metaParameters, ref_model_function=ref_model_function)
    return lambda model_filepath: detector(model_filepath, lr_path, metaParameters, basepath, ref_model_function=ref_model_function)