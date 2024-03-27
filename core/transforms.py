import torch
from sklearn.decomposition import PCA


############## Valid Transform Functions ##############


def sort_all(p):
    return p.reshape(-1).sort()[0]


def pca_eigenvalues(p):
    return pca_feature(p, "eigenvalues")


def pca_tensor(p):
    return pca_feature(p, "tensor")


def pca_components(p):
    return pca_feature(p, "components")


def sv(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        p = p.reshape(p.shape[0], -1)
        return torch.linalg.svd(p, full_matrices=False).S


def sort01(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        return dimlist_sort(p, dimlist=(0, 1))


def sort10(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        return dimlist_sort(p, dimlist=(1, 0))


def sort0(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        return dimlist_sort(p, dimlist=(0,))

def sort1(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        return dimlist_sort(p, dimlist=(1,))
        
def sort_out_in(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        p = p.reshape(p.shape[0], -1)
        return dimlist_sort(p, dimlist=(0, 1))
        
def sort_in_out(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        p = p.reshape(p.shape[0], -1)
        return dimlist_sort(p, dimlist=(1, 0))


############## Common Functions ##############


def pca_feature(tensor, feature_type):
    if tensor.ndim == 1:
        ps = tensor
    elif tensor.ndim >= 2:
        # Flatten the tensor into a 2-D matrix
        tensor = tensor.reshape(tensor.shape[0], -1)
        kd = int(tensor.shape[1]/4.)
        k = min(kd, tensor.shape[0])
        pca = PCA(n_components=k)
        tensor_cpu = tensor.cpu().detach().numpy()
        tensor_pca = pca.fit_transform(tensor_cpu)
        eigenvalues = pca.explained_variance_
        components = pca.components_
        # Convert the PCA result back to a PyTorch tensor and move to CUDA
        original_device = tensor.device
        if feature_type == "eigenvalues":
            pca_feature = torch.tensor(eigenvalues, device=original_device)
        elif feature_type == "tensor":
            pca_feature = torch.tensor(tensor_pca, device=original_device)
        elif feature_type == "components":
            pca_feature = torch.tensor(components, device=original_device)
        else:
            print(f'Wrong feature type given: {feature_type}')
            exit(1)
        ps = pca_feature
    else:
        print("should not be here tensor.ndim {tensor.ndim} ")
        exit(1)
    return ps


# def sort_or_norm(p, normlist=(), sortlist=()):
#
#     p = dimlist_sort(p, dimlist=sortlist)
#     p = dimnorms(p, dimlist=(normlist,), p_norm=2)
#     return p


# def dimnorms(p, dimlist=((0, 1),), p_norm=2):
#     # computes p-norms along each dimension in dims
#     norms = []
#     for d in dimlist:
#         norm = p.norm(dim=d, p=p_norm)
#         norms.append(norm)
#
#     if len(norms)==1:
#         norms = norms[0]
#     else:
#         norms = torch.cat([n.reshape(-1) for n in norms])
#     return norms


# def crit_sort(p, crit, dims=(0, 1)):
#     # computes crit of p along each dim in dims
#     # orders p along that dim according to that ordering
#     # crit: pxdim -> list of indexes with length p.shape[dim] (no?)
#
#     for d in dims:
#         p = crit(p, d)
#
#     return p


def dimlist_sort(p, dimlist=(0, 1)):
    # sorts along dims sequentially.
    # note: if you sort in different order, you get a different result.
    # skip dimensions if n/a
    for d in dimlist:
        if d <= len(p.shape):
            p = p.sort(dim=d)[0]

    return p


