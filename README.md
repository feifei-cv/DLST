# Implementation of Dynamic label smoothing and semantic transport for unsupervised domain adaptation

## Datasets
- [Office-31](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
- [Office-Home](https://www.hemanthdv.org/OfficeHome-Dataset/)

```
Example commands are included in train_DLST.sh.
CUDA_VISIBLE_DEVICES=0 python DLST.py data/office31 -d Office31 -s A -t D -a resnet50 --epochs 15 --seed 1 --log logs1/DLST_dann/Office31/Office31_A2D --base dann

```

## Citation
We adapt our code base from the v0.1 of [the DALIB library](https://github.com/thuml/Transfer-Learning-Library).

## DALIB

> @misc{dalib,  
>  author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},  
>  title = {Transfer-Learning-library},  
>  year = {2020},  
>  publisher = {GitHub},  
>  journal = {GitHub repository},  
>  howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},  
> }  



