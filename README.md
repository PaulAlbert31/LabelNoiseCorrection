# Official PyTorch implementation for the ICML 2019 paper: "Unsupervised Label Noise Modeling and Loss Correction" https://arxiv.org/abs/1904.11238
Training images with correct (green) and incorrect (red) label             |  Network predicitons post training
:-------------------------:|:-------------------------:
![couldn't find image](https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/data/1000before.png)  | ![couldn't find image](https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/data/1000after.png)

You can find in [RunScripts.sh](https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/RunScripts.sh) an example script to run the code for 80% label noise (M-DYR-H and M-DYR-S) and 90% label noise (MD-DYR-SH).

Feel free to modify input parameters for any level of label noise (and other parameters in the paper).

Additionally, the code is now supporting both CIFAR-10 and CIFAR-100 but feel free to adapt it for other datasets such as TinyImagenet by changing the data loaders and modifying the noise addition function in [utils.py](https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py#53)

 | Dependencies  |
| ------------- |
| python == 3.6     |
| pytorch == 0.4.1     |
| cuda92|
| torchvision|
| matplotlib|
| scikit-learn|
| tqdm|

### Environement
If you are using conda, you can execute:
```sh
$ conda env create -f environment.yml
$ conda activate lnoise
```
This will include all dependencies in a new conda environement called lnoise

### Supported datasets:
CIFAR-10 & CIFAR-100 datasets are currently supported and will be downloaded automatically to the path set with --dataset option

### We run our approach on:
CPU: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz GPU: NVIDIA GTX1080Ti

### Parameters details
Execute the following to get details about parameters. Most of them are set by default to replicate our experiments.
``` sh
$ python train.py --h
```

### Accuracies on CIFAR10

|Algorithm\Noise level| |0|20|50|80|90|
|----|----|----|----|----|----|----|
|M-DYR-H|best|**93.6**|**94.0**|**92.0**|**86.8**|40.8|
||last|**93.4**|**93.8**|**91.9**|**86.6**|9.9|
|MD-DYR-SH|best|93.6|93.8|90.6|82.4|**69.1**|
||last|92.7|93.6|90.3|77.8|**68.7**|

### Accuracies on CIFAR100

|Algorithm\Noise level| |0|20|50|80|90|
|----|----|----|----|----|----|----|
|M-DYR-H|best|70.3|68.7|61.7|**48.2**|12.5|
||last|66.2|68.5|58.8|**47.6**|8.6|
|MD-DYR-SH|best|**73.3**|**73.9**|**66.1**|41.6|**24.3**|
||last|**71.3**|**73.34**|**65.4**|35.4|**20.5**|

Accuracies are reported at the end of 300 epochs of training.


Note: We thank authors from [1](https://github.com/facebookresearch/mixup-cifar10) for the mixup and Pytorch implementation of PreAct ResNet (https://github.com/facebookresearch/mixup-cifar10) \
that we use in our code.

[1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz, "mixup: Beyond Empirical Risk Minimization", in International Conference on Learning Representations (ICLR), 2018.

### Please consider citing the following paper if you find this work useful for your research.


```
 @inproceedings{ICML2019_UnsupervisedLabelNoise,
  title = {Unsupervised Label Noise Modeling and Loss Correction},
  authors = {Eric Arazo and Diego Ortego and Paul Albert and Noel E O'Connor and Kevin McGuinness},
  booktitle = {International Conference on Machine Learning (ICML)},
  month = {June},
  year = {2019}
 }
```

Eric Arazo*, Diego Ortego*, Paul Albert, Noel E. O'Connor, Kevin McGuinness, Unsupervised Label Noise Modeling and Loss Correction, International Conference on Machine Learning (ICML), 2019

*Equal contribution
