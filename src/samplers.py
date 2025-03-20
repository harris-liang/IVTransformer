import math
import numpy as np
import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "iv": IVSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class IVSampler(DataSampler):
    def __init__(self, n_dims, noise_std=1, bias=None, scale=None, normalize_Theta=True, mode="linear", q_truncated=None):
        super().__init__(n_dims)
        p = int(n_dims * 1/3)
        q = n_dims - p
        
        self.p = p
        self.q = q
        self.noise_std = noise_std
        self.bias = bias
        self.normalize_Theta = normalize_Theta
        self.mode = mode
        self.q_truncated = q_truncated
        if scale is None:
            self.scale = 1
        else:
            self.scale = scale
        

    def sample_xzs(self, n_points, b_size, U, scale_Theta=1, dist_shift = True, n_dims_truncated=None, seeds=None):
        # scale_Theta: instrumental strength
        zs_b = torch.randn(b_size, n_points, self.q) * self.scale
        Theta_b = torch.randn(b_size, self.q, self.p) # try to fix Theta
        Phi_b = torch.randn(b_size, self.p, self.p)
        E_b = torch.randn(b_size, n_points, self.p) * self.noise_std
        if self.normalize_Theta:
            Theta_b = Theta_b / torch.linalg.norm(Theta_b, dim=(1,2), keepdim=True) * math.sqrt(self.q * self.p)

        if n_dims_truncated is not None or self.q_truncated is not None:
            if n_dims_truncated is not None:
                p_truncated = int(n_dims_truncated * 1/3)
                q_truncated = n_dims_truncated - p_truncated
            if self.q_truncated is not None:
                p_truncated = self.p
                q_truncated = self.q_truncated
    
            # Truncate the data
            zs_b[:, :, q_truncated:] = 0
            U[:, :, p_truncated:] = 0
            E_b[:, :, p_truncated:] = 0

            Theta_b_truncated = torch.zeros_like(Theta_b)
            Theta_b_truncated[:, :q_truncated, :p_truncated] = Theta_b[:, :q_truncated, :p_truncated]
            Phi_b_truncated = torch.zeros_like(Phi_b)
            Phi_b_truncated[:, :p_truncated, :p_truncated] = Phi_b[:, :p_truncated, :p_truncated]
            
            Theta_b = Theta_b_truncated
            Phi_b = Phi_b_truncated
        if self.mode == "linear" or self.mode == "multicollinearity" or self.mode == "multicollinearity-heavy":
            xs_hat_b = scale_Theta * torch.einsum('ijk,ikl->ijl', zs_b, Theta_b)
        elif self.mode == "quadratic":
            xs_hat_b = scale_Theta * torch.einsum('ijk,ikl->ijl', zs_b**2, Theta_b)
        elif self.mode == "non-linear":
        # Using a 2-layer neural network to generate xs_hat_b
            layer_1 = torch.relu(torch.einsum('ijk,ikl->ijl', zs_b, Theta_b))
            Theta_b_2 = torch.randn(b_size, self.p, self.p)
            if self.normalize_Theta:
                Theta_b_2 = Theta_b_2 / torch.linalg.norm(Theta_b_2, dim=(1, 2), keepdim=True) * math.sqrt(self.p * self.p)
            xs_hat_b = scale_Theta * torch.einsum('ijk,ikl->ijl', layer_1, Theta_b_2)

        xs_b = xs_hat_b +  U @ Phi_b + E_b 

        if dist_shift: # test distribution without measurement error
            xs_b[:, -1, :] = xs_hat_b[:, -1, :] + E_b[:, -1, :]

        if self.mode == "multicollinearity":
            zs_b[:, :, -1] = 2 * zs_b[:, :, -2] + torch.randn(b_size, n_points) * 1e-3
            xs_b[:, :, -1] = 2 * xs_b[:, :, -2] + torch.randn(b_size, n_points) * 1e-3
        if self.mode == "multicollinearity-heavy":
            zs_b[:, :, 5:] = 2 * zs_b[:, :, :5] + torch.randn(b_size, n_points, 5) * 1e-3
            xs_b[:, :, 3:] = 2 * xs_b[:, :, 1:3] + torch.randn(b_size, n_points, 2) * 1e-3
        concat_tensor = torch.cat((xs_b, zs_b), dim=2)

        return concat_tensor

     
        
