# MCR2-Guided-Image-Classification
## Introduction
This repository contains the scripts to train and test the models described in the paper "[Hierarchical Training for Distributed Deep Learning Based on Multimedia Data over Band-Limited Networks](https://ieeexplore.ieee.org/document/9897383/)".

### Citation
If you use these scripts in your research, please cite:

    @inproceedings{qi2022hierarchical,
    title={Hierarchical Training for Distributed Deep Learning Based on Multimedia Data over Band-Limited Networks},
    author={Qi, Siyu and Chamain, Lahiru D and Ding, Zhi},
    booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
    pages={2871--2875},
    year={2022},
    organization={IEEE}
    }

## Usage:
```bash
$ cd autoencoder
$ python train_ae.py --arch=resnet18 \ # architecture
                     --data=cifar10 \ # dataset
                     --lcr=0.0 \ # label corruption ratio
                     --lr=0.0001 \ # MCR2 learning rate, should be small
                     --bs=1000 \ # batch size, shall be large enough
                     --epo=0 \ # number of epochs to train Encoder & Decoder, I used 100
                     --DWT \ # apply DWT to input images
                     --pretrain_dir=saved_models/trained_model \ # pretrained model path
                     --model_dir=trained_model \ # directory to save current model
                     --ce_lam=9 # lambda value of CE loss

```
