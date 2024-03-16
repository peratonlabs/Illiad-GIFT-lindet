import os
from core.ref_models import get_refmodelfn
from core.archlindet import ArchLinDet
import joblib
import core.utils
import core.transforms
import torch
import numpy as np


class LinDet:
    def __init__(self, arg_dict, cfg_dicts):
        # base_path = os.path.join(arg_dict['configure_models_dirpath'], 'models')
        # self.base_path = base_path
        self.arg_dict = arg_dict
        self.cfg_dicts = cfg_dicts
        self.gift_basepath = arg_dict['gift_basepath']
        self.scratch_dirpath = arg_dict['scratch_dirpath']

        if arg_dict['learned_parameters_dirpath'] is None:
            self.learned_parameters_dirpath = os.path.join(self.gift_basepath, 'learned_parameters')
        else:
            self.learned_parameters_dirpath = arg_dict['learned_parameters_dirpath']

        self.ref_model_function = self._get_refmodel_fn()
        # self.arch_detectors = {}

    def _get_refmodel_fn(self):
        ref_model_dir = os.path.join(self.gift_basepath, 'reference_models')
        return get_refmodelfn(self.arg_dict['round'], ref_model_dir)

    def cal(self, configure_models_dirpath):

        models_base_path = os.path.join(configure_models_dirpath, 'models')
        modeldirs = os.listdir(models_base_path)
        model_filepaths = [os.path.join(models_base_path, modeldir, 'model.pt') for modeldir in modeldirs]
        cls = [core.utils.get_class_allrounds(os.path.join(models_base_path, modeldir)) for modeldir in modeldirs]

        # holdoutratio = self.arg_dict['cv_test_prop']
        # num_cv_trials = self.arg_dict['num_cv_trials']
        # scratch_dirpath = self.arg_dict['scratch_dirpath']

        # cv_scratch_dir = os.path.join(scratch_dirpath, 'cv_results')
        # os.makedirs(cv_scratch_dir, exist_ok=True)

        arch_map = core.utils.get_archmap(model_filepaths)

        arch_weight_mappings = {}
        arch_classifiers = {}
        print('calibrating for architectures: ', arch_map.keys())
        for arch, arch_inds in arch_map.items():
            print('starting arch', arch)
            cfg_dict = self.cfg_dicts[arch]
            ref_model = self.ref_model_function(arch)
            if torch.cuda.is_available() and ref_model is not None:
                ref_model.cuda()

            archlindet = ArchLinDet(cfg_dict, ref_model, arch=arch)

            arch_fns = np.array([model_filepaths[i] for i in arch_inds])
            arch_classes = np.array([cls[i] for i in arch_inds])

            # if num_cv_trials > 0:
            #     output_path = os.path.join(scratch_dirpath, 'cv_results')
            #     os.makedirs(output_path, exist_ok=True)
            archlindet.run_cv_trials(arch_fns, arch_classes, self.arg_dict)

            arch_weight_mapping = archlindet.train_weight_mapping(arch_fns, arch_classes)
            arch_classifier = archlindet.train_classifier(arch_fns, arch_classes)
            arch_weight_mappings[arch] = arch_weight_mapping
            arch_classifiers[arch] = arch_classifier

        # save trained weight mappings and classifier
        os.makedirs(self.learned_parameters_dirpath, exist_ok=True)
        joblib.dump([arch_weight_mappings, arch_classifiers, self.cfg_dicts],
                    os.path.join(self.learned_parameters_dirpath, 'wa_lr.joblib'))

    def det(self, model_filepath):
        model = torch.load(model_filepath)
        arch = core.utils.get_arch(model)
        # if arch not in self.arch_detectors: #is this worth saving?  could lead to memory issues
        lr_path = os.path.join(self.learned_parameters_dirpath, "wa_lr.joblib")
        mappings, classifiers, xstats_dict = joblib.load(lr_path)

        weight_mapping = mappings[arch]
        classifier = classifiers[arch]
        # xstats = xstats_dict[arch]
        cfg_dict = self.cfg_dicts[arch]
        if self.ref_model_function is None:
            ref_model = None
        else:
            ref_model = self.ref_model_function(arch)
        # self.arch_detectors[arch] = ArchLinDet(cfg_dict, ref_model)
        arch_detector = ArchLinDet(cfg_dict, ref_model, weight_mapping=weight_mapping, classifier=classifier)

        p = arch_detector.detect(model)

        return p


