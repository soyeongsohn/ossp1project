import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

## BrushstrokeOptimizer ##

# num_steps = 100
# style_weight = 3.
# content_weight = 1.
# tv_weight = 0.008
# curv_weight = 4

content_layers = ['conv4_1', 'conv5_1']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

vgg_weight_file = '/content/vgg_weights/vgg19_weights_normalized.h'

def BrushstrokeOptimizer(num_steps=100, style_weight=3., content_weight=1., tv_weight=0.008, curv_weight=4):

  vgg_loss = loss.StyleTransferLosses(vgg_weight_file,content_img, style_img, content_layers, style_layers,scale_by_y=True)
  vgg_loss.to(device).eval() # evaluation 과정에 사용되지 않아야 하는 layer들 off

  optimizer = optim.Adam([location, curve_s, curve_e, curve_c, width], lr=1e-1)
  optimizer_color = optim.Adam([color], lr=1e-2)

  for i in range(num_steps): 
    optimizer.zero_grad()
    optimizer_color.zero_grad()

    content_loss, style_loss = vgg_loss(rendered_image) # rendering된 이미지와 content image, style image의 loss
    content_loss *= content_weight
    style_loss *= style_weight

    curve_loss = loss.curvature_loss(curve_s, curve_e, curve_c)
    curve_loss *= curve_weight
  
    tv_loss = loss.total_variation_loss(location,curve_s,curve_e,K=10) # 입력 이미지의 인접 픽셀 값에 대한 절대 차이로 이미지에 얼마나 많은 노이즈가 있는지 측정
    tv_loss *= tv_weight 

    loss = contetnt_loss + style_loss + curve_loss + tv_loss
    loss.backward(inputs=[location, curve_s, curve_e, curve_c, width], retain_graph=True) # loss function에 적용된 변수들에 대한 기울기 값
    optimizer.step()
    style_loss.backward(inputs=[color])
    optimizer_color.step()

  with torch.no_grad(): # 계산을 비활성화하는 context 관리자
        return rendered_image # optimize된 렌더링 이미지


## PixelOptimizer ##

# num_steps=1000
# style_weight=10000.
# content_weight=1.
# tv_weight=0


def PixelOptimizer(input_img: torch.Tensor, num_steps=1000, style_weight=10000., content_weight=1., tv_weight=0):
  
  input_img = input_img.detach()[None].permute(0, 3, 1, 2).contiguous() # input_img (H,W,3) > (1, 3, H, W)  
  input_img = torch.nn.Parameter(input_img, requires_grad=True) # no_grad()로 false된걸 다시 true

  vgg_loss = loss.StyleTransferLosses(vgg_weight_file, content_image, style_image, content_layers, style_layers)
  vgg_loss.to(device).eval
  optimizer = optim.Adam([input_img], lr=1e-3) # input으로 들어온 optimize된 렌더링 이미지

  for i in range(num_steps):
    optimizer.zero_grad()
    input = torch.clamp(input_img, 0.,1.) # input_img가 image의 범위를 벗어 나지 못하게 clamp함수를 써줌
    content_loss, style_loss = vgg_loss(input)
    content_loss *= content_weight
    style_loss *= style_weight
      
    tv_loss = tv_weight * loss.tv_loss(input_img)

    loss = content_loss + style_loss + tv_loss
    loss.backward(inputs=[input_img])
    optimizer.step()

    return input_img



