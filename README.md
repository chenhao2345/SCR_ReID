# Spatial-Channel Partition Network
Implement of paper:[Learning Discriminative and Generalizable Representations by Spatial-Channel
Partition for Person Re-Identification](http://openaccess.thecvf.com/content_WACV_2020/papers/Chen_Learning_Discriminative_and_Generalizable_Representations_by_Spatial-Channel_Partition_for_Person_WACV_2020_paper.pdf)

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn
- matplotlib
- skimage



## Data

Market1501 data download from [here](http://www.liangzheng.org/Project/project_reid.html)

DukeMTMC-reID data download from [here](http://vision.cs.duke.edu/DukeMTMC/)


## Train

You can specify more parameters in opt.py

```
python3 train_eval.py --mode train --data_path <path/to/Market-1501-v15.09.15> 
```




## Citation

```text
@InProceedings{Chen_2020_WACV,
author = {Chen, Hao and Lagadec, Benoit and Bremond, Francois},
title = {Learning Discriminative and Generalizable Representations by Spatial-Channel Partition for Person Re-Identification},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}
```
