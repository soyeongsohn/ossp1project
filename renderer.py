import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_cluster import knn
from torchvision.transforms import functional

from brushstroke import initialize_brushstrokes


device = "cuda" if torch.cuda.is_available() else "cpu"

def quadratic_bezier_curve(s, e, c, n_points=10):
    
    N = s.shape[0]
    t = torch.linspace(0., 1., n_points, dtype=torch.float32).to(device) #주어진 축을 따라 일정한 간격의 값 생성, t의 범위는 0과 1 사이
    t = torch.stack([t] * N, dim=0)
    
    #각 point들을 x와 y로 분리
    s_x = torch.unsqueeze(s[:, 0], dim=-1)
    s_y = torch.unsqueeze(s[:, 1], dim=-1)

    e_x = torch.unsqueeze(e[:, 0], dim=-1)
    e_y = torch.unsqueeze(e[:, 1], dim=-1)

    c_x = torch.unsqueeze(c[:, 0], dim=-1)
    c_y = torch.unsqueeze(c[:, 1], dim=-1)

#     x = (1. - t) ** 2 * s_x + 2 * (1. - t) * t * c_x + t ** 2 * e_x
#     y = (1. - t) ** 2 * s_y + 2 * (1. - t) * t * c_y + t ** 2 * e_y
    x = c_x + (1. - t) ** 2 * (s_x - c_x) + t ** 2 * (e_x - c_x)
    y = c_y + (1. - t) ** 2 * (s_y - c_y) + t ** 2 * (e_y - c_y)

    return torch.stack([x, y], axis=-1)


def renderer(curve_points, location, color, width, H, W, K=20):

    N, S, _ = curve_points.shape  # brushstroke 수, sample point 수

    # init coordinates tensor (H * W * 2) -> from location
    coordinate_x, coordinate_y = location[:, 0], location[:, 1]

    coordinate_x = torch.unsqueeze(coordinate_x, dim=1)
    coordinate_y = torch.unsqueeze(coordinate_y, dim=1)
    
    coordinate_x = torch.clamp(coordinate_x, 0, W)
    coordinate_y = torch.clamp(coordinate_y, 0, H)
    
    # location으로 다시 합쳐주기
    location = torch.cat([coordinate_x, coordinate_y], dim=-1).to(device) # (6261, 2)
    
    # init tensor of brushstrokes colors
    color = torch.clamp(color, 0, 1) # torch.clamp를 사용하여 상한/하한 값 설정 -> normalize한 color 값 범위(0~1) 안에 들어가도록!
    color = color.to(device)

    # init tensor of brushstorkes widths
    width = width.to(device)

    # create tensor coarse of size H' * W' (H' = H * 0.1, W' = W * 0.1) (Appendix C.2)
    t_W = torch.linspace(0., W, int(W * 0.1), dtype=torch.float32).to(device)
    t_H = torch.linspace(0., H, int(H * 0.1), dtype=torch.float32).to(device)
    # find coordinate
    p_y, p_x = torch.meshgrid(t_H, t_W, indexing='xy')
    p = torch.stack([p_x, p_y], dim=-1) # (25, 25, 2)

    # find k-nearest brushstrokes' indices for every coarse grid cell
    _, indices = knn(location, p.view(-1, 2), k=K) # finds for each element in y tine k nearest points in x
    indices = indices.reshape(len(t_H), len(t_W), -1) # (H', W', K)
    indices = indices.permute(2, 0, 1) # (K, H', W') # resize를 위해서

    # upsampling to H * W
    indices = functional.resize(indices, size=(H, W), interpolation=functional.InterpolationMode.NEAREST)
    # cv2 안 쓴 이유: image를 Pillow로 처리하기 때문, torchvision이 Pillow로 만들었다고 함
    indices = indices.permute(1, 2, 0) # (H, W, K)
    
    # indices 사용하여 k-nearest curve point 좌표 찾기 (논문의 알고리즘을 k-nearest에 대해 적용해야함)
    nearest_Bs = curve_points[indices.flatten()].view(H, W, K, S, 2)

    # init tensor of brushstroke color
    nearest_Bs_colors = color[indices.flatten()].view(H, W, K, 3)

    # init tensor of brushstroke widths
    nearest_Bs_widths = width[indices.flatten()].view(H, W, K, 1)

    # sample points t in [0, 1] for k nearest
    t_H = torch.linspace(0., H, H).to(device)
    t_W = torch.linspace(0., W, W).to(device)
    P_x, P_y = torch.meshgrid(t_W, t_H, indexing='xy')
    p = torch.stack([P_x, P_y], dim=-1)
    
    # assign to every location only the K nearest strokes
    # https://discuss.pytorch.org/t/pytorch-equivalent-of-tf-gather/122058 (논문에서 tf.gather 썼다고 함)
    indices_a = torch.LongTensor([i for i in range(S-1)]).to(device) # 논문처럼 S 사이즈로 하면 다 0됨..
    B_a = nearest_Bs[..., indices_a, :] # start point of each segment
    indices_b = torch.LongTensor([i for i in range(1, S)]).to(device)
    B_b = nearest_Bs[..., indices_b, :] # end point of each segment

    # distances from each sampled point on a stroke to each coordinate, shape = (H, W, K, S)
    B_ba = B_b - B_a
    B_ba = B_ba.permute(1, 0, 2, 3, 4)
    p_B = torch.unsqueeze(torch.unsqueeze(p, axis=2), axis=2) - B_a # (H, W, K, S, 2)
    # t: curve segment 내의 좌표점에 대한 projection
    t = torch.sum(B_ba * p_B, dim=-1) / (torch.sum(B_ba ** 2, dim=-1) + 1e-10) # 아주 작은 수를 더하여 division by zero 에러 방지 
    t = torch.clamp(t, 0, 1)
    nearest_points = B_ba + torch.unsqueeze(t, axis=-1) * B_ba
    dist_nearest_points = torch.sum((torch.unsqueeze(torch.unsqueeze(p, axis=2), axis=2) - nearest_points) ** 2, dim=-1)
    
    # distance from a coordinate x, y to the nearest point on a curve, shape = (H, W, K)
    # tensorflow의 reduce_min이 없고, 동시에 두 차원 못 줄여서 amin 두 번 씀
    D_ = torch.amin(dist_nearest_points, dim=-1) # (H, W, K)
    D = torch.amin(D_, dim=-1) # (H, W) 

    # mask of each stroke, shape = (H, W, K)
    mask = F.softmax(100000. * (1 / (1e-8 + D_)), dim=-1).float()  # (H, W, K)
    # rendering of each stroke, shape = (H, W, 3)
    I_colors = torch.einsum('hwnc,hwn->hwc', nearest_Bs_colors, mask)  # (H, W, 3)
    # assignment, shape = (H, W, N)
    bs = torch.einsum('hwnc,hwn->hwc', nearest_Bs_widths, mask)  # (H, W, 1)
    bs_mask = torch.sigmoid(bs - torch.unsqueeze(D, axis=-1))
    # final rendering
    canvas = torch.ones(I_colors.shape).to(device) * 0.5 # gray
    I = I_colors * bs_mask + (1 - bs_mask) * canvas

    return I  # (H, W, 3)

class Renderer(nn.Module):
    def __init__(self, content_img, H, W, n_strokes=5000, S=10, K=20,
                length_scale=1.1, width_scale=0.1):
        super(Renderer, self).__init__()
        
        self.H = H
        self.W = W
        self.n_strokes = n_strokes
        self.S = S
        self.K = K
        self.length_scale = length_scale
        self.width_scale = width_scale

        location, s, e, c, width, color = initialize_brushstrokes(content_img, n_strokes,
                                                H, W, length_scale, width_scale)
        
        location = location[..., ::-1]
        s = s[..., ::-1]
        e = e[..., ::-1]
        c = c[..., ::-1]

        location = location.astype(np.float32)
        s = s.astype(np.float32)
        e = e.astype(np.float32)
        c = c.astype(np.float32)

        self.curve_s = nn.Parameter(torch.from_numpy(s.copy()), requires_grad=True)
        self.curve_e = nn.Parameter(torch.from_numpy(e.copy()), requires_grad=True)
        self.curve_c = nn.Parameter(torch.from_numpy(c.copy()), requires_grad=True)
        self.color = nn.Parameter(torch.from_numpy(color.copy()), requires_grad=True)
        self.location = nn.Parameter(torch.from_numpy(location.copy()), requires_grad=True)
        self.width = nn.Parameter(torch.from_numpy(width.copy()), requires_grad=True)

    def forward(self):
        curve_points = quadratic_bezier_curve(self.curve_s+self.location, self.curve_e+self.location,
                        self.curve_c+self.location, n_points=self.S)
        canvas = renderer(curve_points, self.location, self.color, self.width,
                self.H, self.W, self.K)
        
        return canvas
