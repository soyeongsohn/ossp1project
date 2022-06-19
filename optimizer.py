# -*- coding: utf-8 -*-
"""optimizer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NDKtAAYHxhch2NvLsYYzCnbd8UR8noyB
"""

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

def optimizer(location, curve_s, curve_e, curve_c, width, color, num_steps, content_weight, style_weight,
              curve_weight, tv_weight):
  # location = torch.tensor([[1., -1.], [1., -1.]])
  # curve_s = torch.tensor([[1., -1.], [1., -1.]])
  # curve_e = torch.tensor([[1., -1.], [1., -1.]])
  # curve_c = torch.tensor([[1., -1.], [1., -1.]])
  # width = torch.tensor([[1., -1.], [1., -1.]])
  # color = torch.tensor([[1., -1.], [1., -1.]])
  # num_steps = 100
  optimizer = optim.Adam([location, curve_s, curve_e, curve_c, width], lr=1e-1)
  optimizer_color = optim.Adam([color], lr=1e-2)

  for i in range(num_steps): 
    optimizer.zero_grad()
    optimizer_color.zero_grad()
    # content_loss = rendering된 이미지와 contetnt image의 loss
    # content_loss *= content_weight
    # style_loss = rendering된 이미지와 style image의 loss
    # style_loss *= style_weight
    # curve_loss *= curve_weight
    # tv_loss *= tv_weight total_variation_loss는 입력 이미지의 인접 픽셀 값에 대한 절대 차이의 합이다. 이것은 이미지에 얼마나 많은 노이즈가 있는지 측정한다.
    # loss = contetnt_loss + style_loss + curve_loss + tv_loss
    # loss.backward(inputs=[location, curve_s, curve_e, curve_c,width], retain_graph=True) # loss function에 적용된 변수들에 대한 기울기 값
    # optimizer.step()
    # style_loss.backward(inputs=[color])
    # optimizer_color.step()
      
    # with torch.no_grad(): gradient 계산을 비활성화한 context 관리자
    #   return bs_renderer() optimize된 렌더링 이미지

#optimize 할 변수
location = torch.tensor([[1., -1.], [1., -1.]])
curve_s = torch.tensor([[1., -1.], [1., -1.]])
curve_e = torch.tensor([[1., -1.], [1., -1.]])
curve_c = torch.tensor([[1., -1.], [1., -1.]])
width = torch.tensor([[1., -1.], [1., -1.]])
color = torch.tensor([[1., -1.], [1., -1.]])
num_steps = 100
optimizer = optim.Adam([location, curve_s, curve_e, curve_c, width], lr=1e-1)
optimizer_color = optim.Adam([color], lr=1e-2)

for i in range(num_steps):
  optimizer.zero_grad()
  optimizer_color.zero_grad()
  # content_loss = rendering된 이미지와 contetnt image의 loss
  # content_loos *= content_weight
  # style_loss = rendering된 이미지와 style image의 loss
  # style_loss *= content_weight
  # curve_loss *= curv_weight
  # tv_loss *= tv_weight total_variation_loss는 입력 이미지의 인접 픽셀 값에 대한 절대 차이의 합이다. 이것은 이미지에 얼마나 많은 노이즈가 있는지 측정한다.
  # loss = contetnt_loss + style_loss + curve_loss + tv_loss
  # loss.backward(inputs=[location, curve_s, curve_e, curve_c,width], retain_graph=True) # loss function에 적용된 변수들에 대한 기울기 값
  # optimizer.step()
  # style_loss.backward(inputs=[color])
  # optimizer_color.step()
      
  # with torch.no_grad(): gradient 계산을 비활성화한 context 관리자
  #   return bs_renderer() optimize된 렌더링 이미지

def optimizer(input_img,num_steps,style_weight,content_weight,tv_weight):
  #input_img = T.nn.Parameter(input_img, requires_grad=True) no_grad()로 false된걸 다시 true

  #optimizer = optim.Adam([input_img], lr=1e-3) input으로 들어온 전에 optimize된 렌더링 이미지

  # for i in range(num_steps):
  #   optimizer.zero_grad()
  #   input = T.clamp(input_img, 0.,1.) 반복 횟수만큼 업데이트를 할건데 이떄 input_img가 image의 범위를 벗어 나지 못하게 clamp함수를 써줌
  #   content_loss = input_img와 content img의 loss
  #   style_loss = input_img와 content img의 loss
  #   tv_loss = tv_weight * tv_loss
  #   loss = content_loss + style_loss + tv_loss
  #   loss.backward(inputs=[input_img])
  #   optimizer.step()

#input_img = T.nn.Parameter(input_img, requires_grad=True) no_grad()로 false된걸 다시 true

#optimizer = optim.Adam([input_img], lr=1e-3) input으로 들어온 전에 optimize된 렌더링 이미지

# for i in range(num_steps):
#   optimizer.zero_grad()
#   input = T.clamp(input_img, 0.,1.) 반복 횟수만큼 업데이트를 할건데 이떄 input_img가 image의 범위를 벗어 나지 못하게 clamp함수를 써줌
#   content_loss = input_img와 content img의 loss
#   style_loss = input_img와 content img의 loss
#   tv_loss = tv_weight * tv_loss
#   loss = content_loss + style_loss + tv_loss
#   loss.backward(inputs=[input_img])
#   optimizer.step()