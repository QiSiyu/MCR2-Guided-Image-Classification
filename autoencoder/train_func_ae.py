import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_architectures(name, dim, im_shape = 32,dropout=False):
    """Returns a network architecture.
    
    Parameters:
        name (str): name of the architecture
        dim (int): feature dimension of vector presentation
    
    Returns:
        net (torch.nn.Module)
        
    """
    _name = name.lower()
    if _name == "resnet18dwt":
        from architectures_ae.resnet_cifar_ae import ResNet18DWT
        net = ResNet18DWT(dim, im_shape=im_shape)
    else:
        raise NameError("{} not found in architectures.".format(name))
    net = torch.nn.DataParallel(net).to(device)
    return net


def load_trainset(name, transform=None, train=True, path="./data/"):
    """Loads a dataset for training and testing. If augmentloader is used, transform should be None.
    
    Parameters:
        name (str): name of the dataset
        transform (torchvision.transform): transform to be applied
        train (bool): load trainset or testset
        path (str): path to dataset base path

    Returns:
        dataset (torch.data.dataset)
    """
    sampler = None
    _name = name.lower()
    if _name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "cifar10"), train=train,
                                                download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "cifar100"), train=train,
                                                 download=True, transform=transform)
        trainset.num_classes = 100
    elif _name == "cifar100coarse":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "cifar100"), train=train,
                                                 download=True, transform=transform)
        trainset.targets = sparse2coarse(trainset.targets) 
        trainset.num_classes = 20
    elif _name == "mnist":
        trainset = torchvision.datasets.MNIST(root=os.path.join(path, "mnist"), train=train, 
                                              download=True, transform=transform)
        trainset.num_classes = 10
    elif _name == "stl10":
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='train', 
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='test', 
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset, sampler
        else:
            trainset.data = np.concatenate([trainset.data, testset.data])
            trainset.labels = trainset.labels.tolist() + testset.labels.tolist()
            trainset.targets = trainset.labels
            return trainset, sampler
    elif _name == "stl10sup":
        trainset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='train', 
                                              transform=transform, download=True)
        testset = torchvision.datasets.STL10(root=os.path.join(path, "stl10"), split='test', 
                                             transform=transform, download=True)
        trainset.num_classes = 10
        testset.num_classes = 10
        if not train:
            return testset, sampler
        else:
            trainset.targets = trainset.labels
            return trainset, sampler
    elif _name == "tiny_imagenet":
        if not train:
            sub_path = "validation"
        else:
            sub_path = "train"
        tiny_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, "tiny_imagenet",sub_path), 
                                             transform=transform)
        targets = torch.tensor(tiny_dataset.targets)
        target_idx = (targets<10).nonzero()
        sampler = torch.utils.data.sampler.SubsetRandomSampler(target_idx)
        return tiny_dataset, sampler

    else:
        raise NameError("{} not found in trainset loader".format(name))
    return trainset, sampler


def load_transforms(name):
    """Load data transformations.
    
    Note:
        - Gaussian Blur is defined at the bottom of this file.
    
    """
    _name = name.lower()
    if _name == "default":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # siyu added
            ])
    elif _name == "cifar":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
    elif _name == "mnist":
         transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]), 
                GaussianBlur(kernel_size=3),
            transforms.ToTensor()])
    elif _name == "stl10":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=9),
            transforms.ToTensor()])
    elif _name == "fashionmnist" or _name == "fmnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomChoice([
                transforms.RandomAffine((-90, 90)),
                transforms.RandomAffine(0, translate=(0.2, 0.4)),
                transforms.RandomAffine(0, scale=(0.8, 1.1)),
                transforms.RandomAffine(0, shear=(-20, 20))]),
            GaussianBlur(kernel_size=3),
            transforms.ToTensor()])
    elif _name == "test":
        transform = transforms.ToTensor()
    elif _name == "tiny_imagenet_test":
        transform = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor()])
    elif _name =="tiny_imagenet":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])
    else:
        raise NameError("{} not found in transform loader".format(name))
    return transform


def load_checkpoint(model_dir, epoch=None, eval_=False, im_shape=32,dropout=False,arc='resnet18'):
    """Load checkpoint from model directory. Checkpoints should be stored in 
    `model_dir/checkpoints/model-epochX.ckpt`, where `X` is the epoch number.
    
    Parameters:
        model_dir (str): path to model directory
        epoch (int): epoch number; set to None for last available epoch
        eval_ (bool): PyTorch evaluation mode. set to True for testing
        
    Returns:
        net (torch.nn.Module): PyTorch checkpoint at `epoch`
        epoch (int): epoch number
    
    """
    if model_dir.endswith('pt'):
        ckpt_path = model_dir
        epoch=0
    elif epoch is None: # get last epoch
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        epoch = np.sort(epochs)[-1]
        ckpt_path = os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch))
        params = utils.load_params(model_dir)
    print('Loading checkpoint: {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    if model_dir.endswith('pt'):
        arc += 'dwt'
        fd = 128
    else:  
        if arc=='resnet18':
            arc = params['arch']
        fd = params['fd']
        if params['DCT'] and (not arc.endswith('DCT')):
            arc += 'dct'
        elif params['DWT'] and (not arc.endswith('DWT')):
            arc += 'dwt'
            
    net = load_architectures(arc,fd ,im_shape,dropout=dropout)
    new_state_dict = {}
    for k in checkpoint['state_dict'].keys():
        new_k = k
        if 'encoder_dense' in k:
            new_k = new_k.replace('encoder_dense','side_branch')
        new_state_dict[new_k] = checkpoint['state_dict'][k]
        
    net.load_state_dict(new_state_dict)
    if eval_:
        net.eval()
    if 'mcr2_optimizer' in checkpoint:
        return net, checkpoint['mcr2_optimizer'], checkpoint['ce_optimizer'], checkpoint['mcr2_scheduler'], checkpoint['ce_scheduler'], epoch
    else:
        return net, None, None, None, None, epoch

def load_checkpoint_1opt(model_dir, epoch=None, eval_=False, im_shape=32,dropout=False):
    """Load checkpoint from model directory. Checkpoints should be stored in 
    `model_dir/checkpoints/model-epochX.ckpt`, where `X` is the epoch number.
    
    Parameters:
        model_dir (str): path to model directory
        epoch (int): epoch number; set to None for last available epoch
        eval_ (bool): PyTorch evaluation mode. set to True for testing
        
    Returns:
        net (torch.nn.Module): PyTorch checkpoint at `epoch`
        epoch (int): epoch number
    
    """
    if model_dir.endswith('pt'):
        ckpt_path = model_dir
        epoch=0
    elif epoch is None: # get last epoch
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        epoch = np.sort(epochs)[-1]
        ckpt_path = os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch))
        params = utils.load_params(model_dir)
    print('Loading checkpoint: {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    if model_dir.endswith('pt'):
        arc = 'resnet18dwt'
        fd = 128
    else:  
        arc = params['arch']
        fd = params['fd']
        if params['DCT'] and (not arc.endswith('DCT')):
            arc += 'dct'
        elif params['DWT'] and (not arc.endswith('DWT')):
            arc += 'dwt'
            
    net = load_architectures(arc,fd ,im_shape,dropout=dropout)
    net.load_state_dict(checkpoint['state_dict'])
    if eval_:
        net.eval()
    return net, checkpoint['mcr2_optimizer'], checkpoint['ce_optimizer'], checkpoint['mcr2_scheduler'], checkpoint['ce_scheduler'], epoch


def load_part_of_model(model_dir, epoch=None, im_shape=32,arc='resnet18'):
    if model_dir.endswith('pt'):
        ckpt_path = model_dir
        epoch=0
    elif epoch is None: # get last epoch
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        epoch = np.sort(epochs)[-1]
        ckpt_path = os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch))
        params = utils.load_params(model_dir)
    print('Loading checkpoint: {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    scheduler = checkpoint['scheduler']
    if model_dir.endswith('pt'):
        if not arc.lower().endswith('dwt'):
            arc += 'dwt'
        fd = 128
    else:
        if arc=='resnet18':
            arc = params['arch']
        fd = params['fd']
        if params['DCT'] and (not arc.lower().endswith('dct')):
            arc += 'dct'
        elif params['DWT'] and (not arc.lower().endswith('dwt')):
            arc += 'dwt'
    print('Architecture %s, dim=%d' % (arc, fd))
    net = load_architectures(arc, fd,im_shape)

    new_state_dict = {}
    for k in checkpoint['state_dict'].keys():
        new_k = k
        if 'encoder_dense' in k:
            new_k = new_k.replace('encoder_dense','side_branch')
        new_state_dict[new_k] = checkpoint['state_dict'][k]
        
    
    added = 0
    for name, param in new_state_dict.items():
        try : 
            net.state_dict()[name].copy_(param)
            added += 1
        except Exception as e:
            print("Didn't load param %s, %s" % (name, e))
            pass

    print('loaded %d%% of params.' % (added / float(len(net.state_dict().keys()))*100))
    return net, checkpoint['optimizer'], scheduler, epoch

def load_prune_model(model_dir, epoch=None, im_shape=32,arc='resnet18'):
    if model_dir.endswith('pt'):
        ckpt_path = model_dir
        epoch=0
    elif epoch is None: # get last epoch
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        epoch = np.sort(epochs)[-1]
        ckpt_path = os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch))
        params = utils.load_params(model_dir)
    print('Loading checkpoint: {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    if model_dir.endswith('pt'):
        arc+='dwt'
        fd = 128
    else:  
        if arc=='resnet18':
            arc = params['arch']
        fd = params['fd']
        if params['DCT'] and (not arc.lower().endswith('dct')):
            arc += 'dct'
        elif params['DWT'] and (not arc.lower().endswith('dwt')):
            arc += 'dwt'
    print('Architecture %s, dim=%d' % (arc, fd))
    net = load_architectures(arc, fd,im_shape)

    added = 0
    for name, param in checkpoint['state_dict'].items():
        try : 
            net.state_dict()[name].copy_(param)
            added += 1
        except Exception as e:
            print("Didn't load param %s, %s" % (name, e))
            pass

    print('loaded %d%% of params.' % (added / float(len(net.state_dict().keys()))*100))
    return net


def get_features(net, trainloader, verbose=True):
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
        batch_features = net(batch_imgs.cuda())
        features.append(batch_features.cpu().detach())
        labels.append(batch_lbls)
    return torch.cat(features), torch.cat(labels)
    

def corrupt_labels(mode="default"):
    """Returns higher corder function"""
    if mode == "default":
        from corrupt import default_corrupt
        return default_corrupt
    elif mode == "asymmetric_noise":
        from corrupt import asymmetric_noise
        return asymmetric_noise
    elif mode == "noisify_pairflip":
        from corrupt import noisify_pairflip
        return noisify_pairflip
    elif mode == "noisify_multiclass_symmetric":
        from corrupt import noisify_multiclass_symmetric
        return noisify_multiclass_symmetric



def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi


def membership_to_label(membership):
    """Turn a membership matrix into a list of labels."""
    _, num_classes, num_samples, _ = membership.shape
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        labels[i] = np.argmax(membership[:, i, i])
    return labels

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


## Additional Augmentations
class GaussianBlur():
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]
