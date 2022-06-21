from PIL import Image
import numpy as np
from skimage.segmentation import slic # color space에서 k-means clustering을 통해 이미지 segment
from scipy.spatial import ConvexHull # 모여있는 점들의 최외곽선을 이어줌



def initialize_brushstrokes(content_img, n_strokes, canvas_h, canvas_w, sec_scale, width_scale):

    # slic: segments image using k-means clustering in color-(x, y, z) space
    segments = slic(
        content_img,
        n_segments=n_strokes,
        min_size_factor=0.02, # Proportion of the minimum segment size to be removed with respect to the supposed segment size `depth*width*height/n_segments`
        max_size_factor=3., # Proportion of the maximum connected segment size. A value of 3 works in most of the cases.
        compactness=2,
        sigma=1, # Width of Gaussian smoothing kernel for pre-processing for each dimension of the image. 
        start_label=1

    )

    # cluster -> strokes, get parameters
    location, s, e, c, width, color = clusters2strokes(segments,
                                                    content_img, 
                                                    canvas_h, 
                                                    canvas_w,
                                                    sec_scale=sec_scale,
                                                    width_scale=width_scale)

    return location, s, e, c, width, color


def clusters2strokes(segments, img, H, W, sec_scale, width_scale):

    n_clusters = np.max(segments) + 1
    cluster_params = {'center': [],
                        's': [],
                        'e': [],
                        'n_pixels': [],
                        'width': [],
                        'rgb': []}

    N = 0

    for cluster_idx in range(1, n_clusters):
        cluster_mask = (segments==cluster_idx)
        if np.sum(cluster_mask) < 5:
            continue

        cluster_mask_nonzeros = np.nonzero(cluster_mask) # 0이 아닌 value의 index
        cluster_points = np.stack((cluster_mask_nonzeros[0], cluster_mask_nonzeros[1]), axis=-1)

        try:
            convex_hull = ConvexHull(cluster_points) # 바깥 노드들만 감싸기(edge)
        except Exception:
            continue
    
        # 클러스터 내에서 가장 거리가 먼 두 점 찾기(width span)
        border_points = cluster_points[convex_hull.simplices.reshape(-1)]
        dist = np.sum((np.expand_dims(border_points, axis=1) - border_points)**2, axis=-1) # border의 각 점에서 다른 점으로의 거리들의 행렬
        max_idx_a, max_idx_b = np.nonzero(dist == np.max(dist)) # 거리의 max값과 같은 값의 인덱스(x, y) 반환
        p_a = border_points[max_idx_a[0]] # 거리가 max가 되는 점의 좌표 1
        p_b = border_points[max_idx_b[0]] # 거리가 max가 되는 점의 좌표 2

        # p_a와 p_b를 이은 직선에 직교하는 직선의 두 교점 찾기
        coef_ba = p_b - p_a # a와 b를 연결한 직선의 방정식 기울기 [분모, 분자]
        # 어떤 직선과 그 직선에 직교하는 직선의 기울기의 곱은 -1이 되어야 함
        if coef_ba[0] != 0:
            orth = np.array([coef_ba[1], -coef_ba[0]])
        else:
            orth = np.array([-coef_ba[1], coef_ba[0]])

        m = (p_a + p_b) / 2 # a와 b를 연결한 직선의 중간값
        p = cluster_points[convex_hull.simplices][:, 0, :] # cluster를 잇는 선의 시작점
        q = cluster_points[convex_hull.simplices][:, 1, :] # cluster를 잇는 선의 끝점
        u =  (orth[0] * (m[1] - p[:, 1]) - orth[1] * (m[0] - p[:, 0])) \
            / (orth[1] * (p[:, 0] - q[:, 0]) - orth[0] * (p[:, 1] - q[:, 1])) 

        intersec_idcs = np.logical_and(u >= 0, u <= 1) # u는 0과 1 사이여야 함
        intersec_points = p + u.reshape(-1, 1) * (q - p)
        intersec_points = intersec_points[intersec_idcs] # m 을 지나는 a, b를 연결한 직선에 수직인 선과 edge의 교점

        width = np.sqrt(np.sum((intersec_points[0] - intersec_points[1]) ** 2))

        if width == 0.: # 너비가 0이면 brushstroke를 형성하지 못함
            continue
        
        N +=1 # 형성된 brushstroke 수

        # 이미지 크기로 정규화하여 파라미터에 추가
        center_x = np.median(cluster_mask_nonzeros[0]) / img.shape[0] # 논문에서는 mean
        center_y = np.median(cluster_mask_nonzeros[1]) / img.shape[1]
        cluster_params["center"].append(np.array([center_x, center_y]))
        cluster_params["s"].append(p_a / img.shape[:2])
        cluster_params["e"].append(p_b / img.shape[:2])
        cluster_params["n_pixels"].append(np.sum(cluster_mask))
        cluster_params["width"].append(width)
        cluster_params["rgb"].append(np.mean(img[cluster_mask], axis=0)) # cluster의 평균 rgb 값 -> 비슷한 것끼리 묶었으므로 평균으로 하더라도 smoothing이 되거나 하지는 않을듯?


    # 모든 cluster parameter들을 numpy array로 변환
    for k in cluster_params.keys():
        cluster_params[k] = np.array(cluster_params[k], dtype=np.float32)
    
    # resolution되었을 때 높이과 너비가 달라지므로 값을 재조정해야 함
    rel_num_pixels = 5 * cluster_params['n_pixels'] / np.sqrt(H * W) # 5를 왜 곱하는지 모르겠다..

    location = cluster_params["center"]
    s = cluster_params["s"]
    e = cluster_params["e"]
    cluster_width = cluster_params["width"]

    # 위치와 관련된 파라미터들은 기존 사이즈로 정규화됨 -> resolution된 높이, 너비를 곱해주어야 함
    location[:, 0] *= H
    location[:, 1] *= W
    s[:, 0] *= H 
    s[:, 1] *= W
    e[:, 0] *= H
    e[:, 1] *= W

    # 각 클러스터 내에서의 좌표 -> location으로 빼서 전체 이미지에 대한 좌표로 변경
    s -= location
    e -= location

    color = cluster_params["rgb"]

    # control point는 시작점과 끝점의 중간에서 random한 값을 더함
    c = (s + e) / 2. + np.stack([np.random.uniform(low=-1, high=1, size=[N]),                                                             
                                np.random.uniform(low=-1, high=1, size=[N])],           
                                axis=-1)
    
    # s, e, c의 중심점
    sec_center = (s + e + c) / 3

    # 각 점에서 중심점을 뺌
    s -= sec_center
    e -= sec_center
    c -= sec_center

    # scale 파라미터로 각각 scaling 해주기
    width = width_scale * rel_num_pixels.reshape(-1, 1) * cluster_width.reshape(-1, 1)
    s *= sec_scale
    e *= sec_scale
    c *= sec_scale

    return location, s, e, c, width, color
