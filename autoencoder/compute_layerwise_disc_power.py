#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
from torch.utils.data import DataLoader

import train_func_ae as tf
import numpy as np

parser = argparse.ArgumentParser(description='Supervised Learning')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')

args = parser.parse_args()


if args.data == "tiny_imagenet":
    im_shape = 64
else:
    im_shape = 32
    
## Load model
assert args.pretrain_dir is not None
net = tf.load_prune_model(args.pretrain_dir, args.pretrain_epo, im_shape=im_shape,arc=args.arch)
net.eval()
for param in net.parameters():
    param.requires_grad = False
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# get test features and labels
if args.data=="tiny_imagenet":
    test_transforms = tf.load_transforms("tiny_imagenet_test")
else:
    test_transforms = tf.load_transforms('test')

# create new trainset loader
trainset,sampler = tf.load_trainset(args.data, test_transforms, train=True, path=args.data_dir)
trainloader = DataLoader(trainset, batch_size=500,sampler=sampler)

testset,sampler = tf.load_trainset(args.data, test_transforms, train=False, path=args.data_dir)
testloader = DataLoader(testset, batch_size=500,sampler=sampler)


# Extractor
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

num_classes=10


# layers to compute disc power
layers = {'conv1':net.module.conv1,
          'layer1.0.conv1': net.module.layer1[0].conv1,
          'layer1.0.conv2': net.module.layer1[0].conv2,
          'layer1.1.conv1': net.module.layer1[1].conv1,
          'layer1.1.conv2': net.module.layer1[1].conv2,
          'layer2.0.conv1': net.module.layer2[0].conv1,
          'layer2.0.conv2': net.module.layer2[0].conv2,
          'layer2.1.conv1': net.module.layer2[1].conv1,
          'layer2.1.conv2': net.module.layer2[1].conv2,
          'bottleneck':net.module.encoder[-1],
          'layer3.0.conv1': net.module.layer3[0].conv1,
          'layer3.0.conv2': net.module.layer3[0].conv2,
          'layer3.1.conv1': net.module.layer3[1].conv1,
          'layer3.1.conv2': net.module.layer3[1].conv2,}


disc_powers = []
print("Training set best filter\t worst\tTest set best\t worst")

for layer_name in layers.keys():    
    x = torch.randn(1, 3, 32, 32)
    handle = layers[layer_name].register_forward_hook(get_activation('conv'))
    output = net(x)
    
    feature_shape = list(activation['conv'].shape[1:])
    num_filters = feature_shape[0]
    
    if len(feature_shape)!=3:
        assert len(feature_shape)==1
        feature_shape += [1,1]
    train_class_avg = torch.zeros((num_classes,feature_shape[0],feature_shape[1]*feature_shape[2]),device=device)
    train_total_avg = torch.zeros((feature_shape[0],feature_shape[1]*feature_shape[2]),device=device)

    test_class_avg = torch.zeros((num_classes,feature_shape[0],feature_shape[1]*feature_shape[2]),device=device)
    test_total_avg = torch.zeros((feature_shape[0],feature_shape[1]*feature_shape[2]),device=device)

    train_class_count = torch.zeros(num_classes,device=device)
    test_class_count = torch.zeros(num_classes,device=device)

    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            features = activation['conv'].view(-1,feature_shape[0],feature_shape[1]*feature_shape[2])
            train_total_avg += torch.mean(features,dim=0)
            for c in range(num_classes):
                of_c = labels == c
                train_class_count[c] += torch.sum(of_c)
                train_class_avg[c] += torch.mean(features[of_c],dim=0)
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            features = activation['conv'].view(-1,feature_shape[0],feature_shape[1]*feature_shape[2])
            test_total_avg += torch.mean(features,dim=0)
            for c in range(num_classes):
                of_c = labels == c
                test_class_count[c] += torch.sum(of_c)
                test_class_avg[c] += torch.mean(features[of_c],dim=0)
                
    train_class_avg /= len(trainloader)
    train_total_avg /= len(trainloader)
    test_class_avg /= len(testloader)
    test_total_avg /= len(testloader)
    
    train_dist = train_class_avg - train_total_avg
    train_Sb = train_class_count[:,None, None, None] * torch.einsum('abcd,abde->abce',train_dist[...,None],train_dist[:,:,None])
    train_Sb = torch.sum(train_Sb,dim=0) # sum over classes
    
    test_dist = test_class_avg - test_total_avg
    test_Sb = test_class_count[:,None, None, None] * torch.einsum('abcd,abde->abce',test_dist[...,None],test_dist[:,:,None])
    test_Sb = torch.sum(test_Sb,dim=0) # sum over classes
    
    # compute discriminability of filters
    train_Sw = torch.zeros((feature_shape[0],feature_shape[1]*feature_shape[2],feature_shape[1]*feature_shape[2]),device=device)
    test_Sw = torch.zeros((feature_shape[0],feature_shape[1]*feature_shape[2],feature_shape[1]*feature_shape[2]),device=device)

    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            features = activation['conv'].view(-1,feature_shape[0],feature_shape[1]*feature_shape[2])
            for f in range(num_filters): # select one filter to compute class-wise mean
                for c in range(num_classes):
                    train_dist = features[labels == c,f] - train_class_avg[c,f]
                    train_Sw[f] += train_dist.t() @ train_dist
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            features = activation['conv'].view(-1,feature_shape[0],feature_shape[1]*feature_shape[2])
            for f in range(num_filters): # select one filter to compute class-wise mean
                for c in range(num_classes):
                    test_dist = features[labels == c,f] - test_class_avg[c,f]
                    test_Sw[f] += test_dist.t() @ test_dist
    
    

    train_D = np.zeros(num_filters)
    test_D = np.zeros(num_filters)
    
    for f in range(num_filters): # select one filter and compute disc power
        train_p = torch.pinverse(train_Sw[f]) @ train_Sb[f]
        train_D[f] = torch.trace(train_p).item()
        test_p = torch.pinverse(test_Sw[f]) @ test_Sb[f]
        test_D[f] = torch.trace(test_p).item()
    train_D_sort_inds = train_D.argsort()
    train_D = train_D[train_D_sort_inds[::-1]]
    test_D = test_D[train_D_sort_inds[::-1]]
    print("%15s %.4f %.4f %.4f %.4f" % (layer_name, train_D.max(), train_D.min(), test_D.max(), test_D.min()))

    handle.remove()
