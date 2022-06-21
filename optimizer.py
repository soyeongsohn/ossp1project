import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

from loss_func import *
from renderer import *

device = "cuda" if torch.cuda.is_available() else "cpu"

## BrushstrokeOptimizer ##
class BrushstrokeOptimizer:
  def __init__(self, content_img, style_img, resolution=512, n_strokes=5000, n_steps=100,
              S=10, K=20, width_scale=0.1, length_scale=1.1, content_weight=1.0, style_weight=3.0,
              tv_weight=0.008, curv_weight=4, streamlit_pbar=None):

    self.resolution = resolution # canvas size
    self.n_strokes = n_strokes
    self.n_steps = n_steps
    self.S = S
    self.K = K
    self.width_scale = width_scale
    self.length_scale = length_scale
    self.content_weight = content_weight
    self.style_weight = style_weight
    self.tv_weight = tv_weight
    self.curv_weight = curv_weight
    self.streamlit_pbar = streamlit_pbar

    W, H = content_img.size

    # H와 W의 비율을 유지하며 resize (긴 부분이 canvas 크기가 되도록)
    if H < W: 
      new_W = resolution
      new_H = int((H / W) * resolution) 
    else:
      new_H = resolution
      new_W = int((W / H) * resolution)

    self.canvas_h = new_H
    self.canvas_w = new_W

    content_img = content_img.resize((self.canvas_w, self.canvas_h))
    style_img = style_img.resize((self.canvas_w, self.canvas_h))

    content_img = torch.tensor(np.array(content_img), dtype=torch.float32).unsqueeze(0).to(device, torch.float32)
    style_img = torch.tensor(np.array(style_img), dtype=torch.float32).unsqueeze(0).to(device, torch.float32)
    content_img = content_img.permute(0, 3, 1, 2)
    style_img = style_img.permute(0, 3, 1, 2)
    # color normalization
    content_img /= 255.
    style_img /= 255.

    self.content_img = content_img
    self.style_img = style_img

    self.content_layers = ['conv4_1', 'conv5_1']
    self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    self.vgg_weight_file = './vgg_weights/vgg19_weights_normalized.h'

  def optimize(self):
    vgg_loss = StyleTransferLosses(self.vgg_weight_file, self.content_img, self.style_img, self.content_layers, self.style_layers, scale_by_y=True)
    vgg_loss.to(device).eval() # evaluation 과정에 사용되지 않아야 하는 layer들 off
    renderer = Renderer(self.content_img[0].permute(1, 2, 0).cpu().numpy(), self.canvas_h, self.canvas_w, self.n_strokes, self.S, self.K, self.length_scale, self.width_scale)
    renderer.to(device)
    optimizer = optim.Adam([renderer.location, renderer.curve_s, renderer.curve_e, renderer.curve_c, renderer.width], lr=1e-1)
    optimizer_color = optim.Adam([renderer.color], lr=1e-2)

    for _ in range(self.n_steps): 
      optimizer.zero_grad()
      optimizer_color.zero_grad()

      canvas = renderer()
      canvas = canvas.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
      content_loss, style_loss = vgg_loss(canvas) # rendering된 이미지와 content image, style image의 loss
      content_loss *= self.content_weight
      style_loss *= self.style_weight

      curve_loss = curvature_loss(renderer.curve_s, renderer.curve_e, renderer.curve_c)
      curve_loss *= self.curv_weight
      
      tv_loss = total_variation_loss(renderer.location, renderer.curve_s, renderer.curve_e,K=self.K) # 입력 이미지의 인접 픽셀 값에 대한 절대 차이로 이미지에 얼마나 많은 노이즈가 있는지 측정
      tv_loss *= self.tv_weight 

      loss = content_loss + style_loss + curve_loss + tv_loss
      loss.backward(inputs=[renderer.location, renderer.curve_s, renderer.curve_e, renderer.curve_c, renderer.width], retain_graph=True) # loss function에 적용된 변수들에 대한 기울기 값
      optimizer.step()
      style_loss.backward(inputs=[renderer.color], retain_graph=True)
      optimizer_color.step()

      if self.streamlit_pbar is not None:
        self.streamlit_pbar.update(1)
      
    with torch.no_grad():
      return renderer()
      
# with torch.no_grad(): gradient 계산을 비활성화한 context 관리자
#   return rendered_image optimize된 렌더링 이미지

## PixelOptimizer ##
class PixelOptimizer:
  def __init__(self, canvas, content_img, style_img, resolution=512, n_steps=1000, 
              style_weight=1000, content_weight=1.0, tv_weight=0.0, streamlit_pbar=None):
    
    self.n_steps = n_steps
    self.style_weight = style_weight
    self.content_weight = content_weight
    self.tv_weight = tv_weight
    self.streamlit_pbar = streamlit_pbar
    self.vgg_weight_file = './vgg_weights/vgg19_weights_normalized.h'
    # input_img (H,W,3) > (1, 3, H, W) 
    self.canvas = canvas
    self.canvas = self.canvas.detach()[None].permute(0, 3, 1, 2).contiguous() 
    self.canvas = torch.nn.Parameter(self.canvas, requires_grad=True) # no_grad()로 false된걸 다시 true
    self.content_layers = ['conv4_1', 'conv5_1']
    self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    W, H = content_img.size

    # H와 W의 비율을 유지하며 resize (긴 부분이 canvas 크기가 되도록)
    if H < W: 
      new_W = resolution
      new_H = int((H / W) * resolution) 
    else:
      new_H = resolution
      new_W = int((W / H) * resolution)

    self.canvas_h = new_H
    self.canvas_w = new_W

    content_img = content_img.resize((self.canvas_w, self.canvas_h))
    style_img = style_img.resize((self.canvas_w, self.canvas_h))

    content_img = torch.tensor(np.array(content_img), dtype=torch.float32).unsqueeze(0).to(device, torch.float32)
    style_img = torch.tensor(np.array(style_img), dtype=torch.float32).unsqueeze(0).to(device, torch.float32)
    content_img = content_img.permute(0, 3, 1, 2)
    style_img = style_img.permute(0, 3, 1, 2)
    # color normalization
    content_img /= 255.
    style_img /= 255.

    self.content_img = content_img
    self.style_img = style_img

  def optimize(self):
    vgg_loss = StyleTransferLosses(self.vgg_weight_file, self.content_img, self.style_img, self.content_layers, self.style_layers)
    optimizer = optim.Adam([self.canvas], lr=1e-3) # input으로 들어온 optimize된 렌더링 이미지

    for _ in range(self.n_steps):
      optimizer.zero_grad()
      input_ = torch.clamp(self.canvas, 0.,1.) # input_img가 image의 범위를 벗어 나지 못하게 clamp함수를 써줌
      content_loss, style_loss = vgg_loss(input_)
      content_loss *= self.content_weight
      style_loss *= self.style_weight
      
      tv_loss_ = self.tv_weight * tv_loss(self.canvas)

      loss = content_loss + style_loss + tv_loss_
      loss.backward(inputs=[self.canvas], retain_graph=True)
      optimizer.step()
  
      if self.streamlit_pbar is not None:
        self.streamlit_pbar.update(1)
    
    return self.canvas


