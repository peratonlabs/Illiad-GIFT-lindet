

import os
import pickle
import numpy as np

import json
import jsonschema
from jsonargparse import ArgumentParser, ActionConfigFile
# import matplotlib.pyplot as plt

import detection.feature_selection
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression


# import warnings
# warnings.filterwarnings("ignore")

# from detection.wa_detection import det
# from detection.wa_detection import cal

# from utils import ref_models
# from utils import schema_ns
# from utils import utils
# import numpy as np
# import os


if __name__ == "__main__":

    
    parser = ArgumentParser()
    parser.add_argument('--cv_dir', type=str)


    args = parser.parse_args()

    print('starting directory', args.cv_dir)

    fns = os.listdir(args.cv_dir)


    archmap = {}
    for fn in fns:
        parts = fn.split('_')
        arch = "_".join(parts[1:-1])
        if arch not in archmap:
            archmap[arch] = []
        archmap[arch].append(fn)

    # nfeats_options = [100,200,300,400,500,700,1000]
    nfeats = 1000
    # C_options = [0.001,0.002,0.003,0.005,0.007,0.01,0.02,0.03,0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3,5,7,10,20,30,50,100]
    C_options = [0.00001,0.0003, 0.0001, 0.0003, 0.001,0.003,0.01,0.03, 0.1, 0.3, 1,  3,10,30,100]


    for arch in archmap.keys():
    
        auc_results = {}
        ce_results = {}

        # for nfeats in nfeats_options:
        #     auc_results[nfeats] = []
        #     ce_results[nfeats] = []
        for C in C_options:
            pred_aucs = []
            pred_ces = []
            for i, fn in enumerate(archmap[arch]):

                with open(os.path.join(args.cv_dir,fn),'rb') as f:
                    stuff = pickle.load(f)

                weight_mapping, classifier, xstats, tr_fns, v_fns, xtr, tr_cls, xv, v_cls = stuff
                # print(xtr.shape)

                # ux, stdx = xstats
                # xtr = (xtr - ux) / stdx
                # xv = (xv - ux) / stdx


                # aucs = detection.feature_selection.get_auc(xtr, tr_cls)

                # aucs = np.abs(aucs.astype(np.float64) - 0.5) + 1E-8 * np.random.randn(*aucs.shape)

                # aucscopy = aucs.copy()
                # aucscopy.sort()
                # thr = aucscopy[-nfeats]

                # xtr = xtr[:,aucs>=thr]
                # xv = xv[:,aucs>=thr]

                classifier = LogisticRegression(max_iter=1000, C=C)

                classifier.fit(xtr, tr_cls)

                pv = classifier.predict_proba(xv)[:, 1]
                try:
                    this_auc = roc_auc_score(v_cls, pv)
                    pred_aucs.append(this_auc)
                except:
                    this_auc = None
                this_ce = log_loss(v_cls, pv, labels=[0,1])

                
                pred_ces.append(this_ce)
                # print('split',i,this_auc,this_ce)
            C_aucs = np.mean(pred_aucs)
            C_ces = np.mean(pred_ces)

            # auc_results[nfeats].append(C_aucs)
            # ce_results[nfeats].append(C_ces)

            print(arch, 'C =',C, ', nfeats =',nfeats,', avg auc:', C_aucs, 'avg ce:', C_ces)








