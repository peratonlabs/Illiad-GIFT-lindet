
import numpy as np
import torch

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