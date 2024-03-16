
import core.utils
import core.transforms
import torch
import numpy as np
import collections
from core.sel_mets import get_auc, get_corr, get_metric_batched
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import os
import pickle

class ArchLinDet:
    def __init__(self, cfg_dict, ref_model, arch=None, weight_mapping=None, classifier=None):
        self.cfg_dict = cfg_dict
        self.arch = arch
        self.weight_mapping = weight_mapping
        self.classifier = classifier

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # maybe add to args?

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

        # if weight_mapping is None:
        #     weight_mapping = self.weight_mapping

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
        # grabs single model's weights and preprocesses according to cfg_dict

        if isinstance(model_or_path, str):
            model = torch.load(model_or_path, map_location=torch.device(self.device))
        else:
            model = model_or_path

        if type(model) == collections.OrderedDict:  # state dictionary
            ps = [v.data.to(self.device) for k, v in model.items() if len(v.shape) > 0]
        else:
            if type(model) == dict: # annoying r16 hack
                import utils.models_r16
                model, model_repr, model_class = utils.models_r16.load_model(model_or_path)
            ps = [mp.data.to(self.device) for mp in model.parameters()]

        if self.ref_model is not None:
            ref_ps = [rp.data.to(self.device) for rp in self.ref_model.parameters()]
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

    def train_weight_mapping(self, model_fns, labels):
        # greedily selects most predictive features

        labels = torch.tensor(labels).to(self.device)

        if self.cfg_dict['ntensors'] == 0:
            weight_mapping = self._train_weight_mapping_simple(model_fns, labels)
        else:
            weight_mapping = self._train_weight_mapping_psel(model_fns, labels)

        self.weight_mapping = weight_mapping

        return weight_mapping

    def train_classifier(self, model_paths, labels):
        # trains a logistic regression classifier on the selected features

        torch.cuda.empty_cache()
        x = [self.get_mapped_weights(fn) for fn in model_paths]
        torch.cuda.empty_cache()
        x = np.stack(x)
        classifier = LogisticRegression(max_iter=1000, C=self.cfg_dict['C'])
        classifier.fit(x, labels)

        self.classifier = classifier
        return classifier

    def _train_weight_mapping_simple(self, model_fns, labels):
        # this treats all weights the same, doesn't care about which tensor they came from

        criterion = self.cfg_dict['feature_selection_criterion']
        param_batch_sz = self.cfg_dict['param_batch_sz']
        nfeats = self.cfg_dict['nfeats']

        ind = 0
        aucs = []
        while True:
            # print('starting param', ind)
            xs = self.get_params(model_fns, ind)
            torch.cuda.empty_cache()  #still important?
            xs_len = len(xs)

            iii = 0
            for x in xs:
                # print('computing aucs',iii)
                iii += 1
                if criterion == 'auc':
                    this_aucs = np.abs(
                        get_metric_batched(x, labels, fun=get_auc, maxelements=200000000).astype(np.float64) - 0.5)
                elif criterion == 'corr':
                    this_aucs = np.abs(
                        get_metric_batched(x, labels, fun=get_corr, maxelements=200000000).astype(np.float64))
                else:
                    assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
                this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)
                aucs.append(this_aucs)
                torch.cuda.empty_cache()

            del xs

            if xs_len < param_batch_sz:
                break

            ind += param_batch_sz

        aucscopy = np.concatenate(aucs)
        aucscopy.sort()
        if aucscopy.shape[0] > nfeats:
            thr = aucscopy[-nfeats]
        else:
            thr = -np.inf

        weight_mapping = []
        for auc in aucs:
            cur_weight_mapping = auc >= thr
            cur_weight_mapping2 = np.where(cur_weight_mapping)[0]

            weight_mapping.append(cur_weight_mapping2)

        torch.cuda.empty_cache()
        return weight_mapping

    def _train_weight_mapping_psel(self, model_fns, labels):
        # this first tries to find the best tensors, then splits selected features evenly among these

        criterion = self.cfg_dict['feature_selection_criterion']
        param_batch_sz = self.cfg_dict['param_batch_sz']
        nfeats = self.cfg_dict['nfeats']
        ntensors = self.cfg_dict['ntensors']

        nfeats_per_tensor = round(nfeats / ntensors)
        pinds = self.select_pinds(model_fns, labels)
        weight_mapping = [[] for i in range(1 + max(pinds))]

        xs = self.get_params(model_fns, -1, pinds=pinds)

        torch.cuda.empty_cache()
        xs_len = len(xs)

        for ii, x in enumerate(xs):
            pind = pinds[ii]

            if criterion == 'auc':
                this_aucs = np.abs(
                    get_metric_batched(x, labels, fun=get_auc, maxelements=200000000).astype(np.float64) - 0.5)
            elif criterion == 'corr':
                this_aucs = np.abs(
                    get_metric_batched(x, labels, fun=get_corr, maxelements=200000000).astype(np.float64))
            else:
                assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
            this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)
            aucscopy = this_aucs.copy()
            aucscopy.sort()
            if aucscopy.shape[0] > nfeats_per_tensor:
                thr = aucscopy[-nfeats_per_tensor]
            else:
                thr = -np.inf
            cur_weight_mapping = this_aucs >= thr
            cur_weight_mapping2 = np.where(cur_weight_mapping)[0]
            weight_mapping[pind] = cur_weight_mapping2
            torch.cuda.empty_cache()

        del xs

        torch.cuda.empty_cache()
        return weight_mapping

    def select_pinds(self, model_fns, labels):
        # tries to find the best tensors

        criterion = self.cfg_dict['feature_selection_criterion']
        param_batch_sz = self.cfg_dict['param_batch_sz']
        ntensors = self.cfg_dict['ntensors']
        nfeats = self.cfg_dict['nfeats']

        ntrials = 10
        holdout_ratio = 0.2

        nfeats_per_tensor = round(nfeats / ntensors)

        ind = 0

        param_aucs = []

        while True:
            xs = self.get_params(model_fns, ind)
            torch.cuda.empty_cache()
            xs_len = len(xs)

            for x in xs:
                this_param_aucs = []
                for trial in range(ntrials):
                    x_tr, x_val, labels_tr, labels_val = core.utils.get_good_split(x, labels, holdout_ratio=holdout_ratio)

                    if criterion == 'auc':
                        this_aucs = np.abs(
                            get_metric_batched(x_tr, labels_tr, fun=get_auc, maxelements=200000000).astype(
                                np.float64) - 0.5)
                    elif criterion == 'corr':
                        this_aucs = np.abs(
                            get_metric_batched(x_tr, labels_tr, fun=get_corr, maxelements=200000000).astype(np.float64))
                    else:
                        assert False, 'invalid criterion' + criterion + ', must be "auc" or "corr"'
                    this_aucs += 1E-8 * np.random.randn(*this_aucs.shape)

                    aucscopy = this_aucs.copy()
                    aucscopy.sort()
                    if aucscopy.shape[0] > nfeats_per_tensor:
                        thr = aucscopy[-nfeats_per_tensor]
                    else:
                        thr = -np.inf
                    cur_weight_mapping = this_aucs >= thr

                    feats_tr = x_tr[:, cur_weight_mapping].detach().cpu().numpy()
                    feats_val = x_val[:, cur_weight_mapping].detach().cpu().numpy()

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

    def get_params(self, model_paths, start_ind, pinds=None):
        # gets subset of preprocessed parameters from all models
        # if pinds is populated, we return exactly those indexes
        # otherwise we return start_ind:start_ind+num_params

        num_params = self.cfg_dict['param_batch_sz']

        output_ps = []
        for model_path in model_paths:
            ps = self.get_deltas(model_path)
            if pinds is None:
                ps = ps[start_ind:start_ind + num_params]
            else:
                ps = [ps[pind] for pind in pinds]
            ps = [p.reshape(-1).cpu() for p in ps]

            if len(output_ps) == 0:
                output_ps = [[p] for p in ps]
            else:
                for i, p in enumerate(ps):
                    if output_ps[i][0].shape[0] == p.shape[0]:
                        output_ps[i].append(p)
                    else:
                        num_params = i
                        output_ps = output_ps[:i]
                        break

        output_ps = [torch.stack(vectors) for vectors in output_ps]

        return output_ps

    def run_cv_trials(self, arch_fns, arch_classes, arg_dict):
        # for educational purposes only

        ns = arch_classes.shape[0]
        arch_name = self.arch

        num_cv_trials = arg_dict['num_cv_trials']
        scratch_dirpath = arg_dict['scratch_dirpath']
        cv_scratch_dir = os.path.join(scratch_dirpath, 'cv_results')
        if num_cv_trials > 0:
            os.makedirs(cv_scratch_dir, exist_ok=True)

        cvcal_scores = []
        truths = []
        for i in range(num_cv_trials):
            tr_fns, v_fns, tr_cls, v_cls = core.utils.get_good_split(arch_fns, arch_classes,
                                                                     holdout_ratio=arg_dict['cv_test_prop'],
                                                                     ignore_val=True)

            cv_det = ArchLinDet(self.cfg_dict, self.ref_model)
            weight_mapping = cv_det.train_weight_mapping(tr_fns, tr_cls)
            classifier = cv_det.train_classifier(tr_fns, tr_cls)

            dump_fn = os.path.join(cv_scratch_dir, 'cvdump' + '_' + arch_name + '_' + str(i) + '.p')

            xtr = [cv_det.get_mapped_weights(fn) for fn in tr_fns]
            xtr = np.stack(xtr)
            xv = [cv_det.get_mapped_weights(fn) for fn in v_fns]
            if len(xv) > 0:
                xv = np.stack(xv)

            xstats = None

            with open(dump_fn, 'wb') as f:
                pickle.dump([weight_mapping, classifier, xstats, tr_fns, v_fns, xtr, tr_cls, xv, v_cls], f)
            pv = [cv_det.detect(fn) for fn in v_fns]
            ce = log_loss(v_cls, pv, labels=[0, 1])
            try:
                print('auc =', roc_auc_score(v_cls, pv), ', ce =', ce)
            except:
                print('auc error (probably due to class balance)', ', ce =', ce)

            cvcal_scores.append(pv)
            truths.append(v_cls)

        # print()

