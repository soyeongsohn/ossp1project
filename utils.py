import torch


def gram_matrix(features):
    """
    to compute style loss
    """
    x, y, z, w = features.size() # x: batch size(=1), b: n_feature_maps, (c, d): dim of feature maps
    features = features.view(x*y, z*w) # (n_feature_maps , dim)

    G = torch.mm(features, features.t())

    return G.div(x*y*z*w)

def projection(z):
    x = z[..., 0]
    y = z[..., 1]

    return torch.stack([x ** 2, y ** 2, x * y], dim=-1)