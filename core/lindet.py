import os
from core.ref_models import get_refmodelfn
import joblib
import core.utils
import core.transforms
import torch
import numpy as np
import collections


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
        pass

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
        arch_detector = ArchLinDet(cfg_dict, ref_model, weight_mapping, classifier)

        p = arch_detector.detect(model)

        return p


class ArchLinDet:
    def __init__(self, cfg_dict, ref_model, weight_mapping, classifier=None):
        self.cfg_dict = cfg_dict
        self.weight_mapping = weight_mapping
        self.classifier = classifier

        norm_type, ref_model = core.utils.proc_feat_type(self.cfg_dict['features'], ref_model)
        self.norm_type = norm_type
        self.ref_model = ref_model

        if cfg_dict['normalize_for_feature_selection']:
            self.post_norm = None
            self.feat_sel_norm = norm_type
        else:
            self.post_norm = norm_type
            self.feat_sel_norm = None

        if cfg_dict['sort_tensors']:
            self.tensor_transform = core.transforms.sort_all
        else:
            self.tensor_transform = None

    def detect(self, model_or_path):
        # Runs the detector on this model, returns P(trojan)
        assert self.classifier is not None, "Untrained ArchLitDet trying to classify"
        x = [self.get_mapped_weights(model_or_path)]
        p = self.classifier.predict_proba(x)[:, 1][0]
        return p

    def get_mapped_weights(self, model_or_path):
        # grabs all mapped weights (i.e., selected features) for this model
        assert self.weight_mapping is not None, "ArchLitDet trying to get mapped weights without map"

        # get the weights
        with torch.no_grad():
            ps = self.get_deltas(model_or_path)

        # build into a single vector
        mapped_weights = []
        for i in range(len(self.weight_mapping)):
            param = (ps[i]).cpu().detach().numpy()
            param = param.reshape(-1)
            mapped_weights.append(param[self.weight_mapping[i]])
        mapped_weights = np.concatenate(mapped_weights)

        # normalize feature vector if needed
        if self.post_norm == 'cosine':
            mapped_weights = mapped_weights * np.sqrt(mapped_weights.shape[0]) / np.linalg.norm(mapped_weights)
        elif self.post_norm == 'white':
            mapped_weights = mapped_weights / mapped_weights.std()
        else:
            assert self.post_norm is None, "bad normalization option"

        return mapped_weights

    def get_deltas(self, model_or_path):
        # grabs model weights and preprocesses according to cfg_dict
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if isinstance(model_or_path, str):
            model = torch.load(model_or_path, map_location=torch.device(device))
        else:
            model = model_or_path

        if type(model) == collections.OrderedDict:  # state dictionary
            ps = [v.data.to(device) for k, v in model.items() if len(v.shape) > 0]
        else:
            if type(model) == dict: # annoying r16 hack
                import utils.models_r16
                model, model_repr, model_class = utils.models_r16.load_model(model_or_path)
            ps = [mp.data.to(device) for mp in model.parameters()]

        if self.ref_model is not None:
            ref_ps = [rp.data.to(device) for rp in self.ref_model.parameters()]
            final_ps = []
            for p, ref_p in zip(ps, ref_ps):
                if p.shape == ref_p.shape:
                    final_ps.append(p.data - ref_p.data)
            ps = final_ps

        norm = self.feat_sel_norm
        if norm is not None:
            if norm == 'pnorm':
                ps_new = []
                for p in ps:
                    std = p.std()
                    if std == 0 or std.isnan().any():
                        ps_new.append(p)
                    else:
                        ps_new.append(p / std)
                ps = ps_new
            else:
                if norm == 'white':
                    scaling_factor = torch.std(torch.cat([p.reshape(-1) for p in ps]))
                elif norm == 'cosine':
                    vec = torch.cat([p.reshape(-1) for p in ps])
                    scaling_factor = torch.norm(vec) / np.sqrt(vec.shape[0])
                else:
                    scaling_factor = norm
                ps = [p / scaling_factor for p in ps]

        if self.tensor_transform is not None:
            ps = [self.tensor_transform(p) for p in ps]

        return ps

