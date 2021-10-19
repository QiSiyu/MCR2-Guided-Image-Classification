import argparse
import os
import sys
if not os.path.abspath('..') in sys.path:
    sys.path.append(os.path.abspath('..'))

import numpy as np
from torch.utils.data import DataLoader
from augmentloader import AugmentLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch

import train_func_ae as tf
from loss import MaximalCodingRateReduction
import utils
import time


parser = argparse.ArgumentParser(description='Supervised Learning')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='architecture for deep neural network (default: resnet18)')
parser.add_argument('--fd', type=int, default=128,
                    help='dimension of feature dimension (default: 128)')
parser.add_argument('--data', type=str, default='cifar10',
                    help='dataset for training (default: CIFAR10)')
parser.add_argument('--epo', type=int, default=100,
                    help='number of epochs for training (default: 100)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--lr', type=float, default=0.0003,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gam1', type=float, default=1.,
                    help='gamma1 for tuning empirical loss (default: 1.)')
parser.add_argument('--gam2', type=float, default=0.5,
                    help='gamma2 for tuning empirical loss (default: 0.5)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared (default: 0.5)')
parser.add_argument('--corrupt', type=str, default="default",
                    help='corruption mode. See corrupt.py for details. (default: default)')
parser.add_argument('--lcr', type=float, default=0.,
                    help='label corruption ratio (default: 0)')
parser.add_argument('--lcs', type=int, default=10,
                    help='label corruption seed for index randomization (default: 10)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')
parser.add_argument('--transform', type=str, default='default',
                    help='transform applied to trainset (default: default')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help='base directory for saving PyTorch model. (default: ./data/)')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=None,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--DWT', '-w', action='store_false',
                    help='apply DWT to inputs')
parser.add_argument('--model_dir', type=str, default=None,
                    help='Name of folder for saving PyTorch model. (default: None)')
parser.add_argument('--ce_lam', type=float, default=9.,
                    help='Weight for cross entropy loss (default: 9.)')
parser.add_argument('--ce_model_name', type=str, default='ce_model.pt',
                    help='Name of cross entropy model')

args = parser.parse_args()

print('Weight of cross entropy loss: ', args.ce_lam)

## Pipelines Setup
if args.model_dir is None:
    model_dir = os.path.join(args.save_dir,
               'sup_{}+{}_{}_epo{}_bs{}_lr{}_mom{}_wd{}_gam1{}_gam2{}_eps{}_lcr{}{}'.format(
                    args.arch, args.fd, args.data, args.epo, args.bs, args.lr, args.mom,
                    args.wd, args.gam1, args.gam2, args.eps, args.lcr, args.tail))
else:
    model_dir = os.path.join(args.save_dir,args.model_dir)
utils.init_pipeline(model_dir,resume_training=(args.pretrain_dir is not None and (not args.pretrain_dir.endswith('pt')) ))


im_shape = 32
test_transforms = tf.load_transforms('test')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Prepare for Training
if args.DWT:
    args.arch+='DWT'
    print('Apply DWT to inputs')

print('Using separate optimizers for MCR2 and CE loss, both updating part of model')
ce_param_list = ['decoder', 'layer3', 'final_lin']


if args.pretrain_dir is not None:
    # load pretrained model
    pure_ce_epoch = 0

    if not (args.pretrain_dir).endswith('pt'):
        net, mcr2_optimizer_state, ce_optimizer_state, mcr2_scheduler, ce_scheduler, pretrained_epoch = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo, im_shape=im_shape)
        utils.update_params(model_dir, args.pretrain_dir)
    else:
        net, _, loaded_scheduler, pretrained_epoch = tf.load_part_of_model(args.pretrain_dir, args.pretrain_epo, im_shape=im_shape,arc=args.arch)

    mcr2_params = [p for n, p in net.named_parameters() if (not any(ce_param in n for ce_param in ce_param_list))]
    ce_params = [p for n, p in net.named_parameters() if any(ce_param in n for ce_param in ce_param_list)]
    mcr2_optimizer = optim.SGD(mcr2_params, lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    ce_optimizer = optim.SGD(ce_params, lr=args.lr, momentum=args.mom, weight_decay=args.wd)

    # if not (args.pretrain_dir).endswith('pt'):
    #      mcr2_optimizer.load_state_dict(mcr2_optimizer_state)
    #      ce_optimizer.load_state_dict(ce_optimizer_state)
    # else:
    mcr2_scheduler = lr_scheduler.MultiStepLR(mcr2_optimizer, [args.epo//4, args.epo//2, (3*args.epo)//4], gamma=0.1)
    ce_scheduler = lr_scheduler.MultiStepLR(ce_optimizer, [args.epo//4, args.epo//2, (3*args.epo)//4], gamma=0.1)

else:
    # initialize new model
    pretrained_epoch = 0
    net = tf.load_architectures(args.arch, args.fd, im_shape)
    pure_ce_epoch = 200
    mcr2_params = [p for n, p in net.named_parameters() if (not any(ce_param in n for ce_param in ce_param_list))]
    ce_params = [p for n, p in net.named_parameters() if any(ce_param in n for ce_param in ce_param_list)]
    mcr2_optimizer = optim.SGD(mcr2_params, lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    ce_optimizer = optim.SGD(ce_params, lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    mcr2_scheduler = lr_scheduler.MultiStepLR(mcr2_optimizer, [args.epo//4, args.epo//2, (3*args.epo)//4], gamma=0.1)
    ce_scheduler = lr_scheduler.MultiStepLR(ce_optimizer, [args.epo//4, args.epo//2, (3*args.epo)//4], gamma=0.1)



transforms = tf.load_transforms(args.transform)
trainset,sampler = tf.load_trainset(args.data, transforms, path=args.data_dir)

print('corrupt label ratio = %.2f' % args.lcr)
trainset = tf.corrupt_labels(args.corrupt)(trainset, args.lcr, args.lcs)

assert (sampler is None) or (args.lcr==0), "Can't sample a subset of classes and corrupt classes together!"
trainloader = DataLoader(trainset, batch_size=args.bs, drop_last=True, shuffle=True, num_workers=4,sampler=sampler)
ce_trainloader = DataLoader(trainset, batch_size=200, drop_last=True, shuffle=True, num_workers=4,sampler=sampler)

testset,sampler = tf.load_trainset(args.data, test_transforms, train=False, path=args.data_dir)
testloader = DataLoader(testset, batch_size=200,sampler=sampler, num_workers=4)


mcr2_criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
ce_criterion = torch.nn.CrossEntropyLoss()
mse_criterion = torch.nn.MSELoss()

utils.save_params(model_dir, vars(args))

num_classes = trainset.num_classes


## pre-train with CE
print('Pretrain model for %d epochs with CE loss' % pure_ce_epoch)
pretrain_optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=args.mom, weight_decay=args.wd)
milestone_percentage = [0.5, 0.75, 0.9]
pretrain_scheduler = lr_scheduler.MultiStepLR(pretrain_optimizer, milestones=list(map(lambda x: int(x*pure_ce_epoch),milestone_percentage)), gamma=0.1)

for epoch in range(pure_ce_epoch):
    start = time.time()
    net.train()
    running_ce_loss = 0.0
    running_mse_loss = 0.0

    for step, (batch_imgs, batch_lbls) in enumerate(ce_trainloader):
        pretrain_optimizer.zero_grad()
        output = net(batch_imgs.to(device))
        loss = ce_criterion(output[-1], batch_lbls.long().to(device))
        mse_loss = mse_criterion(output[0],output[-2])
        running_ce_loss += loss.item()
        running_mse_loss += mse_loss.item()
        loss.backward()
        pretrain_optimizer.step()
        _, predicted = output[-1].max(1)

        if step == len(ce_trainloader)-1:
            net.eval()
            total_train = 0
            correct_train = 0
            with torch.no_grad():
                for data in ce_trainloader:
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

            print('[Epoch %d] time elapsed: %.1f seconds, ce loss: %.3f, mse loss: %.3f, train_acc: %.2f%%, val acc: %.2f%%' % \
                  (epoch + 1,time.time()-start,
                   running_ce_loss / len(ce_trainloader),
                   running_mse_loss / len(ce_trainloader),
                   100 * correct_train / total_train,
                   100 * correct_test / total_test))
        del loss
    pretrain_scheduler.step()
    if args.pretrain_dir is None:
        torch.save({
          'state_dict': net.state_dict(),
          'optimizer' : pretrain_optimizer.state_dict(),
           'scheduler': pretrain_scheduler,
        },  os.path.join(args.save_dir, args.ce_model_name))
print("Cross entropy training finished.")



## Hierarchical training
best_acc = 0.
best_mcr2 = 0.
for epoch in range(pretrained_epoch, args.epo):
    start = time.time()
    net.train()
    running_loss = 0.0
    running_mcr2_loss = 0.0
    running_mse_loss = 0.0
    running_latent_bits = 0.0
    for step, (batch_imgs, batch_lbls) in enumerate(trainloader):
        _, latent_bits, features, _, _ = net(batch_imgs.to(device))
        # phase 1: mcr2 backprop
        mcr2_loss, loss_empi, loss_theo = mcr2_criterion(features, batch_lbls, num_classes=num_classes)

        utils.save_state(model_dir, epoch, step, mcr2_loss.item(), *loss_empi, *loss_theo)
        running_mcr2_loss += mcr2_loss.item()

        running_latent_bits += latent_bits.sum().item()
        mcr2_optimizer.zero_grad()
        mcr2_loss.backward()
        mcr2_optimizer.step()

        # phase 2: ce backprop
        orig, latent, features, recon, logits = net(batch_imgs.to(device))
        mcr2_loss, loss_empi, loss_theo = mcr2_criterion(features, batch_lbls, num_classes=num_classes)
        ce_loss = args.ce_lam * ce_criterion(logits, batch_lbls.long().to(device))
        running_loss += ce_loss.item()

        ce_optimizer.zero_grad()
        ce_loss.backward()
        ce_optimizer.step()

    mcr2_scheduler.step()
    ce_scheduler.step()

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

    print('[Epoch %d] time elapsed: %.1f seconds, latent bits: %.3f, mcr2 loss: %.3f, avg ce loss: %.3f, avg mse loss: %.3f, train acc: %.2f%%, val acc: %.2f%%' % \
          (epoch + 1,
           time.time()-start,
           running_latent_bits/len(trainloader),
           running_mcr2_loss/len(trainloader),
           running_loss / len(trainloader),
           running_mse_loss / len(trainloader),
           100 * correct_train / total_train, 100 * correct_test / total_test))


    del mcr2_loss, ce_loss, loss_empi, loss_theo
    del features, logits
    if (correct_test / total_test) > best_acc or running_mcr2_loss/len(trainloader) < best_mcr2:
        print('Saving model...')
        utils.save_ckpt_2opt(model_dir, net, mcr2_optimizer,ce_optimizer, mcr2_scheduler, ce_scheduler, epoch)
        best_acc = (correct_test / total_test)
        best_mcr2 = running_mcr2_loss/len(trainloader)
print("training complete.")
