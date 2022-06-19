import os
import requests
import torch
import torch.nn as nn
import h5py
import numpy as np
import torch.nn.functional as F
import pickle
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

def download_file(url, name):
  if not os.path.exists(name):
    filedir, filename = os.path.split(name)
    os.makedirs(filedir, exist_ok = True)
    response = requests.get(url, stream=True)
    ckpt_file = name 
    with open(ckpt_file, 'wb') as file:
      for data in response.iter_content(chunk_size=1024):
        file.write(data)

download_file('https://github.com/ftokarev/tf-vgg-weights/raw/master/vgg19_weights_normalized.h5',
                                'vgg_weights/vgg19_weights_normalized.h')

file = h5py.File('/content/vgg_weights/vgg19_weights_normalized.h', mode='r')

# 이미지 정규화
mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class ConvRelu(torch.nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'
                 ):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()

class VGG(torch.nn.Sequential):
  def __init__(self, param_file):
    super(VGG, self).__init__()
    self.f = h5py.File(param_file, mode='r')
    self.normalization = Normalization(mean, std)

    self.conv1_1 = ConvRelu(3, 64)
    self.conv1_2 = ConvRelu(64, 64)
    self.pool1 = nn.MaxPool2d(2, stride=2)

    self.conv2_1 = ConvRelu(64, 128)
    self.conv2_2 = ConvRelu(128, 128)
    self.pool2 = nn.MaxPool2d(2, stride=2)

    self.conv3_1 = ConvRelu(128, 256)
    self.conv3_2 = ConvRelu(256, 256)
    self.conv3_3 = ConvRelu(256, 256)
    self.conv3_4 = ConvRelu(256, 256)
    self.pool3 = nn.MaxPool2d(2, stride=2)

    self.conv4_1 = ConvRelu(256, 512)
    self.conv4_2 = ConvRelu(512, 512)
    self.conv4_3 = ConvRelu(512, 512)
    self.conv4_4 = ConvRelu(512, 512)
    self.pool4 = nn.MaxPool2d(2, stride=2)

    self.conv5_1 = ConvRelu(512, 512)
    self.conv5_2 = ConvRelu(512, 512)
    self.conv5_3 = ConvRelu(512, 512)
    self.conv5_4 = ConvRelu(512, 512)
    self.pool5 = nn.MaxPool2d(2, stride=2)
    
    #parameter가 training 되지 않게.
    for p in self.parameters():
            p.requires_grad_(False)

    self.load_params()

    
  def load_params(self):
        trained = [np.array(layer[1], 'float32') for layer in list(self.f.items())]
        weight_value_tuples = []
        for p, tp in zip(self.parameters(), trained):
            if len(tp.shape) == 4:
                tp = np.transpose(tp, (3, 2, 0, 1))
            weight_value_tuples.append((p, tp))

        paramvalues = zip(*(weight_value_tuples))

        #parameter 위치에 대응하는 pretrained된 값 입력 
        for p, v in zip(*paramvalues):
          p.data.copy_(torch.from_numpy(v).data)

        
  def extra_features(self, x):
    features ={}
    x = self.normalization(x)
    x = self.conv1_1(x)
    features['conv1_1'] = x
    x = self.conv1_2(x)
    features['conv1_2'] = x
    x = self.pool1(x)
    features['conv1_2_pool'] = x

    x = self.conv2_1(x)
    features['conv2_1'] = x
    x = self.conv2_2(x)
    features['conv2_2'] = x
    x = self.pool2(x)
    features['conv2_2_pool'] = x
    
    x = self.conv3_1(x)
    features['conv3_1'] = x
    x = self.conv3_2(x)
    features['conv3_2'] = x
    x = self.conv3_3(x)
    features['conv3_3'] = x
    x = self.conv3_4(x)
    features['conv3_4'] = x
    x = self.pool3(x)
    features['conv3_4_pool'] = x

    x = self.conv4_1(x)
    features['conv4_1'] = x
    x = self.conv4_2(x)
    features['conv4_2'] = x
    x = self.conv4_3(x)
    features['conv4_3'] = x
    x = self.conv4_4(x)
    features['conv4_4'] = x
    x = self.pool4(x)
    features['conv4_4_pool'] = x

    x = self.conv5_1(x)
    features['conv5_1'] = x
    x = self.conv5_2(x)
    features['conv5_2'] = x
    x = self.conv5_3(x)
    features['conv5_3'] = x
    x = self.conv5_4(x)
    features['conv5_4'] = x
    x = self.pool5(x)
    features['conv5_4_pool'] = x

    return features
