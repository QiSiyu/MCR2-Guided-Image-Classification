#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
if not os.path.abspath('..') in sys.path:
    sys.path.append(os.path.abspath('..'))
   
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from collections import defaultdict
import train_func_ae as tf
import utils
import numpy as np
import torch.nn.utils.prune as prune
from tqdm import tqdm
import torchac
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


parser = argparse.ArgumentParser(description='Supervised Learning')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--eval', action='store_true',
                    help='Replace denser layers to evaluate')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--corrupt', type=str, default="default",
                    help='corruption mode. See corrupt.py for details. (default: default)')
parser.add_argument('--lcr', type=float, default=0.,
                    help='label corruption ratio (default: 0)')
parser.add_argument('--lcs', type=int, default=10,
                    help='label corruption seed for index randomization (default: 10)')

args = parser.parse_args()


im_shape = 32
num_classes=10

## Prepare for Training
assert args.pretrain_dir is not None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# create new trainset loader
test_transforms = tf.load_transforms('test')
transforms = tf.load_transforms('default')
trainset,sampler = tf.load_trainset(args.data, transforms, path=args.data_dir)
if args.lcr > 0:
    print('corrupt label ratio = %.2f' % args.lcr)
    trainset = tf.corrupt_labels(args.corrupt)(trainset, args.lcr, args.lcs)
trainloader = DataLoader(trainset, batch_size=200, drop_last=True, shuffle=True, num_workers=4,sampler=sampler)

testset,sampler = tf.load_trainset(args.data, test_transforms, train=False, path=args.data_dir)
testloader = DataLoader(testset, batch_size=200,sampler=sampler, num_workers=4)

# Extractor
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook




def compute_disc_power(net, layer, amount=0.5):
    '''Compute pruning mask based on discriminative power in layer. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        layer (torch.nn.Linear): layer to be pruned
        amount (float): a fraction number between 0~1, indicating how much filters/nodes to be pruned

    Returns:
        mask (torch.tensor): pruning mask for layer

    '''    
    device = next(net.parameters()).device
    net.eval()
    x = torch.zeros(1, 3, 32, 32)
    handle = layer.register_forward_hook(get_activation('intermediate_out'))
    _ = net(x)
    
    feature_shape = list(activation['intermediate_out'].shape[1:])
    num_filters = feature_shape[0]
    
    if len(feature_shape)!=3:
        assert len(feature_shape)==1
        feature_shape += [1,1]
    train_class_avg = torch.zeros((num_classes,feature_shape[0],feature_shape[1]*feature_shape[2]),device=device)
    train_total_avg = torch.zeros((feature_shape[0],feature_shape[1]*feature_shape[2]),device=device)
    train_class_count = torch.zeros(num_classes,device=device)

    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            _ = net(images)
            features = activation['intermediate_out'].view(-1,feature_shape[0],feature_shape[1]*feature_shape[2])
            train_total_avg += torch.mean(features,dim=0)
            for c in range(num_classes):  # the ten classes
                of_c = labels == c
                train_class_count[c] += torch.sum(of_c)
                train_class_avg[c] += torch.mean(features[of_c],dim=0)
     
    train_class_avg /= len(trainloader)
    train_total_avg /= len(trainloader)
    
    train_dist = train_class_avg - train_total_avg # shape = (class, filter, feature_dim)
    train_Sb = train_class_count[:,None, None, None] * torch.einsum('abcd,abde->abce',train_dist[...,None],train_dist[:,:,None])
    # # check results
    train_Sb = torch.sum(train_Sb,dim=0) # sum over classes
    # compute discriminability of filters
    train_Sw = torch.zeros((feature_shape[0],feature_shape[1]*feature_shape[2],feature_shape[1]*feature_shape[2]),device=device)

    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            _ = net(images)
            features = activation['intermediate_out'].view(-1,feature_shape[0],feature_shape[1]*feature_shape[2])
            for f in range(num_filters): # select one filter
                for c in range(num_classes): # class-wise mean
                    train_dist = features[labels == c,f] - train_class_avg[c,f]
                    train_Sw[f] += train_dist.t() @ train_dist
  
    # Sw and Sb shape: (filter, feature_dim, feature_dim)
    train_D = np.zeros(num_filters)
    
    for f in range(num_filters): # select one filter
        train_p = torch.pinverse(train_Sw[f]) @ train_Sb[f]
        train_D[f] = torch.trace(train_p).item()
    handle.remove()
    
    disc_scores = torch.from_numpy(train_D).to(device)
    sorted_scores, _ = torch.sort(disc_scores)
    score_threshold = sorted_scores[max(0,amount-1)]
    print('Threshold: ', score_threshold)
    if len(layer.weight.shape) == 2:
        return torch.ones(layer.weight.shape,device=device) * (disc_scores > score_threshold).view(-1, 1)
    return torch.ones(layer.weight.shape,device=device) * (disc_scores > score_threshold).view(-1,1,1,1)

def get_features(net, trainloader, hook, verbose=True):
    '''Extract all features out into one single batch. 
    
    Parameters:
        net (torch.nn.Module): get features using this model
        trainloader (torchvision.dataloader): dataloader for loading data
        verbose (bool): shows loading staus bar

    Returns:
        features (torch.tensor): with dimension (num_samples, feature_dimension)
        labels (torch.tensor): with dimension (num_samples, )
    '''
    features = []
    labels = []
    if verbose:
        train_bar = tqdm(trainloader, desc="extracting all features from dataset")
    else:
        train_bar = trainloader
    for step, (batch_imgs, batch_lbls) in enumerate(train_bar):
        _ = net(batch_imgs.cuda())
        features.append(hook['intermediate_out'].cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)

def entropy_coder(net, layer, trainloader,testloader,activation):
    '''Compute bit rate using arithmetic encoder. 
        
    Parameters:
        net (torch.nn.Module): get features using this model
        layer (torch.nn.Linear): bottleneck layer in net
        trainloader (torchvision.dataloader): dataloader for collecting CDF
        testloader (torchvision.dataloader): dataloader for bit rate evaluation
        activation (dict): used to extract intermediate features

    Returns:
        bitrate (float): average bit rate of output from "layer".
        '''
    net.eval()
    x = torch.zeros(1, 3, 32, 32)
    handle = layer.register_forward_hook(get_activation('intermediate_out'))
    _ = net(x)
    train_loss= defaultdict(float)
    test_loss= defaultdict(float)
    feature_shape = list(activation['intermediate_out'].shape[1:])
    num_filters = feature_shape[0]
    avg_mean = 0
    avg_std = 0
    min_val = torch.zeros([1],device='cuda')
    max_val = 0
    real_bits = 0
    for step, (batch_imgs, batch_lbls) in enumerate(trainloader):
        _ = net(batch_imgs.cuda())
        features = torch.round(activation['intermediate_out'])#.view(-1,num_filters*feature_shape[1]*feature_shape[2]))
        

        avg_mean += torch.mean(features, dim = 0,keepdim=True)
        avg_std += torch.std(features, dim = 0,keepdim=True)
        cur_min, _ = features.min(dim=0,keepdim=True)
        min_val = torch.min(min_val, cur_min)
        max_val = max(max_val, torch.max(features))

    avg_mean /= (step+1)
    avg_mean -= min_val
    avg_std /= (step+1)
    max_val += 1
    
    estimated_dist = torch.distributions.normal.Normal(loc=avg_mean.unsqueeze(-1), scale=avg_std.unsqueeze(-1))
    if len(feature_shape) == 1:
        output_cdf = estimated_dist.cdf(torch.arange(0,max_val,device='cuda')).repeat(testloader.batch_size,1,1)
    else:
        output_cdf = estimated_dist.cdf(torch.arange(0,max_val,device='cuda')).repeat(testloader.batch_size,1,1,1,1)

    for step, (batch_imgs, batch_lbls) in enumerate(testloader):
        _ = net(batch_imgs.cuda())
        features = torch.round(activation['intermediate_out'])#.view(-1,num_filters*feature_shape[1]*feature_shape[2]))
        
        if torch.max(torch.round(features-min_val)) >= max_val-1:
            max_val = torch.max(torch.round(features-min_val))+2
            # print('update max value to ', max_val)
            # print('actual max val = ', torch.max((features-min_val).short()))
            if len(feature_shape) == 1:
                output_cdf = estimated_dist.cdf(torch.arange(0,max_val,device='cuda')).repeat(testloader.batch_size,1,1)
            else:
                output_cdf = estimated_dist.cdf(torch.arange(0,max_val,device='cuda')).repeat(testloader.batch_size,1,1,1,1)
            # print('output_cdf shape ', output_cdf.shape)

        byte_stream = torchac.encode_float_cdf(output_cdf.to('cpu'), (features-min_val).short().to('cpu'), check_input_bounds=True)
        real_bits += len(byte_stream) * 8

    handle.remove()

    print('test bpp = %.9f' % (real_bits/10000/32/32))
    return (real_bits/10000/32/32)
    




def test(net,trainloader,testloader):
    net.eval()
    # print accuracy
    total_train = 0
    correct_train = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs[-1].data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels.to(device)).sum().item()
            
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs[-1].data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels.to(device)).sum().item()
            
    print('Train_acc: %.2f%%, val acc: %.2f%%' %  (100 * correct_train / total_train, 100 * correct_test / total_test))
    return correct_test / total_test

milestone_percentage = [0.5, 0.75, 0.9]
ce_criterion = torch.nn.CrossEntropyLoss()
    

# percentage amount to prune bottleneck layer
prune_num = np.concatenate(((np.arange(0,1,0.1)*128).astype(int),
                            np.arange(int(128*.9)+1, 128)))

bpp = []
cnn_acc = []
for amount in prune_num:
    ## Reload model
    if args.pretrain_dir is not None:
        pure_ce_epoch = 10
    
        if not (args.pretrain_dir).endswith('pt'):
            net, _, _, _, _, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo, im_shape=im_shape,arc=args.arch)
        else:
            net, _, _, _ = tf.load_part_of_model(args.pretrain_dir, 
                                                 args.pretrain_epo, 
                                                 im_shape=im_shape,
                                                 arc=args.arch)

    bottleneck_layer = net.module.encoder[-1]
    
    last_layer_params = [p for n, p in net.named_parameters() if 'final_lin' in n]
    print("Pruning %d filters from " % amount, bottleneck_layer)
    prune.custom_from_mask(bottleneck_layer, name='weight', mask=compute_disc_power(net, bottleneck_layer, amount))
    prune.remove(bottleneck_layer, 'weight')

    # evaluate rate
    bpp.append(np.round(entropy_coder(net, bottleneck_layer, trainloader,testloader,activation),decimals=4))
    
    
    for n, param in net.named_parameters():
        if 'final_lin' not in n:
            param.requires_grad = False

    # compute initial accuracy for comparison with fine-tuned accuracy
    best_acc = test(net,trainloader,testloader)


    # tune final dense layer for "pure_ce_epoch" epochs
    print('Finetune final dense layer for %d epochs with CE loss' % pure_ce_epoch)
    pretrain_optimizer = optim.SGD(last_layer_params, lr=0.00001, momentum=0.9, weight_decay=5e-4)
    milestone_percentage = [0.5, 0.75, 0.9]
    pretrain_scheduler = lr_scheduler.MultiStepLR(pretrain_optimizer, milestones=list(map(lambda x: int(x*pure_ce_epoch),milestone_percentage)), gamma=0.1)
    
    for epoch in range(pure_ce_epoch):
        start = time.time()
        net.train()
        running_loss = 0.0
        best_acc = 0.
        for step, (batch_imgs, batch_lbls) in enumerate(trainloader):
            pretrain_optimizer.zero_grad()
            output = net(batch_imgs.to(device))
            loss = ce_criterion(output[-1], batch_lbls.long().to(device))
            loss.backward()
            pretrain_optimizer.step()
            running_loss += loss.item()
            _, predicted = output[-1].max(1)
    
            if step == len(trainloader)-1:
                net.eval()
                total_train = 0
                correct_train = 0
                with torch.no_grad():
                    for data in trainloader:
                        images, labels = data
                        outputs = net(images.to(device))
                        _, predicted = torch.max(outputs[-1].data, 1)
                        total_train += labels.size(0)
                        correct_train += (predicted == labels.to(device)).sum().item()
                
                correct_test = 0
                total_test = 0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        outputs = net(images.to(device))
                        _, predicted = torch.max(outputs[-1].data, 1)
                        total_test += labels.size(0)
                        correct_test += (predicted == labels.to(device)).sum().item()
                best_acc = max(correct_test / total_test, best_acc)
            del loss
        pretrain_scheduler.step()
    print("Cross entropy training complete, best accuracy = ",best_acc)
    cnn_acc.append(best_acc)

    print('-'*50)
    
print('Printing bpp and CNN accuracy')
print(bpp)
print(cnn_acc)
