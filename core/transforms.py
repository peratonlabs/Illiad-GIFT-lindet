
import torch
from sklearn.decomposition import PCA

def sort_all(p):
    return p.reshape(-1).sort()[0]


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


def pca_eigenvalues(p):
    return pca_feature(p, "eigenvalues")


def pca_tensor(p):
    return pca_feature(p, "tensor")


def pca_components(p):
    return pca_feature(p, "components")

