# MCR2-Guided-Image-Classification
Usage:
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
                     --ce_lam=9 # lambda value of CE loss

```
