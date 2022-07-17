#!/usr/bin/env bash

### Office31 dann 
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_dann/Office31/Office31_A2D --base dann
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_dann/Office31/Office31_A2W --base dann
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_dann/Office31/Office31_D2A --base dann
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_dann/Office31/Office31_W2A --base dann
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 5 --seed 1 --log logs1/DLST_dann/Office31/Office31_D2W --base dann
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 5 --seed 1 --log logs1/DLST_dann/Office31/Office31_W2D --base dann

### Office31 cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_cdan/Office31/Office31_A2D --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s A -t W -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_cdan/Office31/Office31_A2W --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s D -t A -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_cdan/Office31/Office31_D2A --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s W -t A -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_cdann/Office31/Office31_W2A --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s D -t W -a resnet50 --epochs 5 --seed 1 --log logs1/DLST_cdan/Office31/Office31_D2W --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s W -t D -a resnet50 --epochs 5 --seed 1 --log logs1/DLST_cdan/Office31/Office31_W2D --base cdan

### Office-Home
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Ar2Cl --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Ar -t Pr -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Ar2Pr --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Ar -t Rw -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Ar2Rw --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Cl -t Ar -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Cl2Ar --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Cl -t Pr -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Cl2Pr --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Cl -t Rw -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Cl2Rw --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Pr -t Ar -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Pr2Ar --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Pr -t Cl -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Pr2Cl --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Pr -t Rw -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Pr2Rw --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Rw -t Ar -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Rw2Ar --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Rw -t Cl -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Rw2Cl --base cdan
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/office-home -d OfficeHome -s Rw -t Pr -a resnet50 --epochs 25 --seed 1 --log logs1/lst_cdan/OfficeHome/OfficeHome_Rw2Pr --base cdan


### VisDA-2017
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101  --epochs 25 --seed 2 --per-class-eval --center-crop --log logs1/lst_cdan/VisDA2017 --base cdan --trade-off 1.0 --trade-off1 0.2 --trade-off2 0.01

## VisDA-2017
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/visda-2017 -d VisDA2017 -s Synthetic -t Real -a resnet101  --epochs 25 --seed 1 --per-class-eval --center-crop --log logs/DLST_dann/VisDA2017 --base dann --trade-off 1.0 --trade-off1 0.2 --trade-off2 0.01




## DomainNet
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s c -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_c2i --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s c -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_c2p --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s c -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_c2r --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s c -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_c2s --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s i -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_i2c --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s i -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_i2p --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s i -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_i2r --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s i -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_i2s --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s p -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_p2c --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s p -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_p2i --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s p -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_p2r --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s p -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_p2s --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s r -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_r2c --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s r -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_r2i --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s r -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_r2p --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s r -t s -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_r2s --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s s -t c -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_s2c --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s s -t i -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_s2i --base cdan --lr 0.002
##CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s s -t p -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_s2p --base cdan --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python DLST.py data/domainnet -d DomainNet -s s -t r -a resnet101 --epochs 30 -i 2500 -p 500 --seed 0 --log logs1/lst_cdan/DomainNet/DomainNet_s2r --base cdan --lr 0.002