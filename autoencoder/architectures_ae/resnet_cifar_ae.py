#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

################ Functions for DWT #################

def rgb2ycbcr(im):
    transform_kernel = torch.from_numpy(np.array([[.299, .587, .114], 
                                                  [-.1687, -.3313, .5],
                                                  [.5, -.4187, -.0813]],
                                                 dtype=np.float32)).float().to(device)
    offset_kernel = torch.FloatTensor([0,128,128]).view(3,1,1).to(device)
    ycbcr = torch.einsum('adbc,de -> aebc',im, transform_kernel)
    ycbcr += offset_kernel
    return ycbcr

def getTcdf53(height):
    a1 = -0.5
    a2 = 0.25
    a3 = np.sqrt(2.0)
    a4 = 1.0/a3
    X1 = np.identity(height)
    X2 = np.identity(height)
    X3 = np.zeros((height,height)).astype('float32')
    for col in range(1,height-2,2):
        X1[col-1,col]=X1[col+1,col]=a1
    X1[height-2,height-1] = 2*a1
    
    #print(X1)
    for col in range(2,height-1,2):
        X2[col-1,col]=X2[col+1,col]=a2
    X2[1,0] = 2*a2
    #print(X2)
    for col in range(0,height,1):
        if(col%2==0 ):
            #print(col)
            X3[col,int(col/2)]=a3
        else:
            X3[col,int(height/2 + (col-1)/2)]=a4
    #print(X3)
    X =np.matmul(np.matmul(X1,X2),X3)
    return X,np.linalg.inv(X)

def batch_dwt_cdf(M, im_shape=32,h_w_c_dim=[2,3,1]):
    h_dim, w_dim, c_dim = h_w_c_dim
    if c_dim == 1:
        einsum_str1 = 'ij,abjk->abik'
        einsum_str2 = 'abij, jk->abik'
    elif c_dim in [-1,3]:
        einsum_str1 = 'ij,ajkb->aikb'
        einsum_str2 = 'aijb, jk->aikb'
    else:
        print("Current implementation doesn't support channel dimension %d!" % h_w_c_dim[-1])
    def dwt(inputs):
        inputs = torch.einsum(einsum_str1, M.T, inputs)
        inputs = torch.einsum(einsum_str2, inputs, M)
        inputs = inputs.unfold(h_dim, 16, 16).unfold(w_dim, 16, 16)
        inputs = inputs.permute(0, c_dim, h_dim, w_dim, 4, 5)
        inputs = inputs.contiguous().view(inputs.size()[0], -1, *inputs.size()[-2:])
        return inputs
    return dwt






################ Functions for ResNet #################


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        if type(planes)==int:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
    
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )
        else:
            self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes[0])
            self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes[1])
    
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes[-1],
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes[-1])
                )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            return (x + torch.rand(x.shape,device=x.device)*2 - 1)
        else:
            return torch.round(x)




class ResNetDWT(nn.Module):
    def __init__(self, block, num_blocks, feature_dim=512,im_shape=32):
        super(ResNetDWT, self).__init__()
        print('Building ResNet-18-based AE for DWT, with MCR2 side branch...')

        self.in_planes = 64
        self.layer_width = 64
        self.feature_dim = feature_dim
        self.im_shape = im_shape
        input_dim = 12
        block_expansion = 2
           
        self.bn0 = nn.BatchNorm2d(input_dim)
        self.eps=1e-7

        # ResNet modules
        self.conv1 = nn.Conv2d(input_dim, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # M1
        self.layer1 = self._make_layer(block, self.layer_width, num_blocks[0], stride=1)
        
        # M2
        m2_width = int(self.layer_width*block_expansion)
        self.layer2 = self._make_layer(block, m2_width, num_blocks[1], stride=2)
        
        # M3
        self.layer3 = self._make_layer(block, int(self.layer_width*block_expansion**2), num_blocks[2], stride=2)


        # feature map encoder & decoder
        encoder_width = 512
        self.encoder = torch.nn.Sequential(
            nn.Linear(m2_width, encoder_width, bias=False), # bias was false
            nn.BatchNorm1d(encoder_width),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_width, feature_dim, bias=True)
        )
        self.quantize = Quantizer()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(m2_width, m2_width, 2, stride=2),  # b, 128, 2, 2
            nn.ReLU(True),
            nn.ConvTranspose2d(m2_width, m2_width, 2, stride=2, padding=0),  # b, 128, 4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(m2_width, m2_width, 2, stride=2, padding=0),  # b, 128, 8, 8
        )


        # MCR2 side branch
        self.side_branch = torch.nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim, bias=True)
        )
        

        # DWT functions
        M,_ = getTcdf53(im_shape)
        self.dwt = batch_dwt_cdf(nn.Parameter(torch.from_numpy(M.astype(np.float32)).to(device)))
        
        # final dense layer
        self.final_lin = nn.Linear(int(self.layer_width*block_expansion**2 * block.expansion), 10, bias=True)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        '''Returns:
            Input to feature map encoder
            Estimated rate of latent representation (used only for observation)
            Output of MCR2 side branch
            Output of feature map decoder
            Predicted logits
            '''
        x = self.dwt(rgb2ycbcr(x*255.-128))/255

        x = self.bn1(self.conv1(x))
        x = self.layer1(F.relu(x))
        feature_map = self.layer2(x)

        mcr2_out = F.avg_pool2d(feature_map, feature_map.size(2))
        encoded = self.encoder(mcr2_out.view(mcr2_out.size(0), -1))
        mcr2_out = self.quantize(encoded)
        
        recon = self.decoder(mcr2_out.unsqueeze(-1).unsqueeze(-1))

        
        out = self.layer3(recon)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.final_lin(out)
        return  feature_map,\
                torch.log(1e-7+torch.abs(self.quantize(encoded).view(encoded.size(0), -1)))/torch.log(torch.ones(1,device=device)*2), \
                F.normalize(self.side_branch(mcr2_out)),\
                recon, \
                out



def ResNet18DWT(feature_dim=10, im_shape=32):
    return ResNetDWT(BasicBlock, [2, 2, 2, 2], feature_dim,im_shape)




