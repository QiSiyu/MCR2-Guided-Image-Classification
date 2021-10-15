import os
import logging
import json
import numpy as np
import torch
from pathlib import Path


def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def init_pipeline(model_dir, headers=None, resume_training=False):
    """Initialize folder and .csv logger."""
    ## siyu commented out
    # # project folder
    # os.makedirs(model_dir)
    # os.makedirs(os.path.join(model_dir, 'checkpoints'))
    # os.makedirs(os.path.join(model_dir, 'figures'))
    # os.makedirs(os.path.join(model_dir, 'plabels'))
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(model_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(model_dir, 'figures')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(model_dir, 'plabels')).mkdir(parents=True, exist_ok=True)


    if headers is None:
        headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e", 
            "discrimn_loss_t",  "compress_loss_t"]
    create_csv(model_dir, 'losses.csv', headers,resume_training)
    print("project dir: {}".format(model_dir))

def create_csv(model_dir, filename, headers, resume_training=False):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path) and (not resume_training):
        os.remove(csv_path)
    if not resume_training:
        with open(csv_path, 'w+') as f:
            f.write(','.join(map(str, headers)))
    return csv_path

def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def update_params(model_dir, pretrain_dir):
    """Updates architecture and feature dimension from pretrain directory 
    to new directoy. """
    old_params = load_params(pretrain_dir)
    if model_dir is not None:
        params = load_params(model_dir)
        params['arch'] = old_params["arch"]
        params['fd'] = old_params['fd']
        save_params(model_dir, params)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def save_state(model_dir, *entries, filename='losses.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def save_ckpt(model_dir, net, optimizer, scheduler, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    try:
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        last_epoch = np.sort(epochs)[-1]
        os.remove(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(last_epoch)))
    except:
        pass
    # if os.path.exists(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch-1))):  # checking if there is a file with this name
    #     os.remove(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch-1)))
    torch.save({
      'state_dict': net.state_dict(),
      'optimizer' : optimizer.state_dict(),
       'scheduler': scheduler,
    },  os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch)))

def save_ckpt_2opt(model_dir, net, mcr2_optimizer, ce_optimizer, mcr2_scheduler, ce_scheduler, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    try:
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        epochs = [int(e[11:-3]) for e in os.listdir(ckpt_dir) if e[-3:] == ".pt"]
        last_epoch = np.sort(epochs)[-1]
        os.remove(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(last_epoch)))
    except:
        pass
    # if os.path.exists(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch-1))):  # checking if there is a file with this name
    #     os.remove(os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch-1)))
    torch.save({
      'state_dict': net.state_dict(),
      'mcr2_optimizer' : mcr2_optimizer.state_dict(),
      'ce_optimizer' : ce_optimizer.state_dict(),
      'mcr2_scheduler': mcr2_scheduler,
      'ce_scheduler': ce_scheduler,
    },  os.path.join(model_dir, 'checkpoints', 'model-epoch{}.pt'.format(epoch)))


def save_labels(model_dir, labels, epoch):
    """Save labels of a certain epoch to directory. """
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def clustering_accuracy(labels_true, labels_pred):
    """Compute clustering accuracy."""
    from sklearn.metrics.cluster import supervised
    from scipy.optimize import linear_sum_assignment
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)

