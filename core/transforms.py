
import torch
from sklearn.decomposition import PCA

############## Valid Transform Functions ##############

def sort_all(p):
    return p.reshape(-1).sort()[0]

def pca_eigenvalues(p):
    return pca_feature(p, "eigenvalues")


def pca_rightsingular(p):
    return pca_feature(p, "rightsingular")


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



def sv_sortall(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        p = p.reshape(p.shape[0], -1)
        
        U, S, Vh = torch.linalg.svd(p, full_matrices=False)
        
        #left = U @ torch.diag(S)
        #right = torch.diag(S) @ Vh
        
        left = U
        right = Vh
        
        left = left.sort(dim=0)[0].reshape(-1)
        right = right.sort(dim=1)[0].reshape(-1)
        

        return torch.cat([left, right])
        #return left
        #return right





def sv_sort(p):
    if len(p.shape) <= 1:
        return 0*p.reshape(-1).sort()[0]
    else:
        p = p.reshape(p.shape[0], -1)
        
        U, S, Vh = torch.linalg.svd(p, full_matrices=False)
        
        left = U @ torch.diag(S)
        right = torch.diag(S) @ Vh
        #rank = 5
        
        #left = U[:,:rank]
        #right = Vh[:rank]
        
        left_rows = U[:,0].sort()[1]
        right_cols = Vh[0,:].sort()[1]
        
        left = left[left_rows]
        right = right[:,right_cols]
        
        
        left = left.reshape(-1)
        right = right.reshape(-1)
        

        return torch.cat([left, right])


def sort_sv(p):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:
        p = p.reshape(p.shape[0], -1)
        
        p0 = sort0(p)
        p1 = sort1(p)
        
        
        p=sort01(p)
        #return p
        rank=5
        
        U, S0, Vhxx = torch.linalg.svd(p0, full_matrices=False)
        Uxx, S1, Vh = torch.linalg.svd(p1, full_matrices=False)
        
        #S0 = S0[:rank]
        #U = U[:,:rank]
        #S1 = S1[:rank]
        #Vh = Vh[:rank]
        
        left = U @ torch.diag(S0)
        right = torch.diag(S1) @ Vh
        #rank = 5
        
        #left = U
        #right = Vh
        
        #left_rows = U[:,0].sort()[1]
        #right_cols = Vh[0,:].sort()[1]
        
        #left = left[left_rows]
        #right = right[:,right_cols]
        
        
        left = left.reshape(-1)
        right = right.reshape(-1)
        

        return torch.cat([left, right])


def sv_sort_recombine_rank1(p):
    return sv_sort_recombine(p, rank=1)
def sv_sort_recombine_rank5(p):
    return sv_sort_recombine(p, rank=5)
def sv_sort_recombine_4d_rank1(p):
    return sv_sort_recombine_4d(p, rank=1)
def sv_sort_recombine_4d_rank5(p):
    return sv_sort_recombine_4d(p, rank=5)



def sv_sort_recombine(p, rank=100000000):
    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    else:

        p = p.reshape(p.shape[0], -1)
        
        U, S, Vh = torch.linalg.svd(p, full_matrices=False)
        
        #left = U @ torch.diag(S)
        #right = torch.diag(S) @ Vh
        #rank = 1000
        
        #S[rank:]=0
        
        #left = U[:,:rank]
        #right = Vh[:rank]
        
        left = U
        right = Vh
        
        
        #left_rows = (U @ torch.diag(S)).mean(dim=1).sort()[1]
        #right_cols = (torch.diag(S) @ Vh).mean(dim=0).sort()[1]
        
        left_rows = (U @ torch.diag(S))[:,:rank].mean(dim=1).sort()[1]
        right_cols = (torch.diag(S) @ Vh)[:rank,:].mean(dim=0).sort()[1]
        
        #left_rows = U[:,:rank].mean(dim=1).sort()[1]
        #right_cols = Vh[:rank,:].mean(dim=0).sort()[1]
        
        #left_rows = U[:,0].sort()[1]
        #right_cols = Vh[0,:].sort()[1]
        
        left = left[left_rows]
        right = right[:,right_cols]
        #print(left.shape, S.shape,right.shape)
        
        #pp=left @ torch.diag(S)
        #ppp=pp @ right
        
        
        final = left @ torch.diag(S) @ right
        
        
        #left = left.reshape(-1)
        #right = right.reshape(-1)
        

        return final



def sv_sort_recombine_4d(p, rank=100000000):

    if len(p.shape) <= 1:
        return p.reshape(-1).sort()[0]
    elif len(p.shape) == 2:
        U, S, Vh = torch.linalg.svd(p, full_matrices=False)
        
        left_rows = (U @ torch.diag(S))[:,:rank].mean(dim=1).sort()[1]
        right_cols = (torch.diag(S) @ Vh)[:rank,:].mean(dim=0).sort()[1]
        left = U[left_rows]
        right = Vh[:,right_cols]
        final = left @ torch.diag(S) @ right
        
        return final

    else:
        assert len(p.shape) == 4
        
        orig_shape = p.shape

        p = p.reshape(p.shape[0], -1)
        U, S, Vh = torch.linalg.svd(p, full_matrices=False)
        
        left_rows = (U @ torch.diag(S))[:,:rank].mean(dim=1).sort()[1]
        left = U[left_rows]
        
        
        SVh_4d=(torch.diag(S) @ Vh)[:rank,:].reshape(-1, *orig_shape[1:])

        
        right_cols = SVh_4d.mean(dim=3).mean(dim=2).mean(dim=0).sort()[1]
        

        Vh_4d = Vh.reshape(-1, *orig_shape[1:])
        
        Vh_4d = Vh_4d[:,right_cols]
        
        
        right = Vh_4d.reshape(Vh_4d.shape[0],-1)
        
        
        
        
        #left_rows = U[:,:rank].mean(dim=1).sort()[1]
        #right_cols = Vh[:rank,:].mean(dim=0).sort()[1]
        
        #left_rows = U[:,0].sort()[1]
        #right_cols = Vh[0,:].sort()[1]
        
        #left = left[left_rows]
        #right = right[:,right_cols]
        #print(left.shape, S.shape,right.shape)
        
        #pp=left @ torch.diag(S)
        #ppp=pp @ right
        
        
        final = left @ torch.diag(S) @ right
        
        
        #left = left.reshape(-1)
        #right = right.reshape(-1)
        

        return final

#def sv_sort_recombine_4d(p):
#    fgdfgdf

############## Common Functions ##############

    
def effective_dims(tensor):
    return sum(d > 1 for d in tensor.shape)

def pca_feature(tensor, feature_type):
    tensor = tensor.squeeze() # remove all dimensions of size 1
    #ndim = effective_dims(tensor)
    if tensor.ndim == 1:
        ps = tensor
    elif tensor.ndim >= 2:
        # Flatten the tensor into a 2-D matrix
        tensor = tensor.reshape(tensor.shape[0], -1)
        kd = int(tensor.shape[1]/4.)
        k = min(kd, tensor.shape[0])
        U, S, V = torch.pca_lowrank(tensor, q=k)
        #U: contains the left singular vectors (principal components) of the input data matrix. These are the eigenvectors of 
        #   the covariance matrix of the dataset, which represent the directions of maximum variance in the data.
        #S: is a diagonal matrix containing the singular values, which are related to the eigenvalues of the covariance matrix of the dataset.
        #   These values quantify the amount of variance captured by each principal component.
        #V: contains the right singular vectors, which represent the coefficients for each principal component, loadings matrix,
        #   providing a basis for transforming the original data into the principal component space.
        rightsingular = V
        eigenvalues = S
        components = U
        if feature_type == "eigenvalues":
            pca_feature = eigenvalues
        elif feature_type == "rightsingular":
            pca_feature = rightsingular
        elif feature_type == "components":
            pca_feature = components
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


