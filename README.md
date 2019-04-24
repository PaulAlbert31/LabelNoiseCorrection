# Official PyTorch implementation for the ICML 2019 paper: "Unsupervised label noise modeling and loss correction"
Original corrupted images used for training             |  Network prediciton post training
:-------------------------:|:-------------------------:
![couldn't find image](https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/data/1000before.png)  | ![couldn't find image](https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/data/1000after.png)

You can find in https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/RunScript.sh an example script to run the code for 80% label noise (M-DYR-H and M-DYR-S) and 90% label noise (MD-DYR-SH).

Feel free to modify input parameters for any level of label noise (and other parameters in the paper).

Additionally, the code is now supporting both CIFAR-10 and CIFAR-100 but feel free to adapt it for other datasets such as TinyImagenet by changing the data loaders and modifying the noise addition function in https://github.com/PaulAlbert31/LabelNoiseCorrection/blob/master/utils.py#53

 | Dependencies  |
| ------------- |
| python == 3.6     |
| pytorch == 0.4.1     |
| numpy|
| scipy|

### Supported datasets:
  CIFAR-10 & CIFAR-100 datasets - python version, will be downloaded automatically to the path set with --dataset option

### We run our approach on:
CPU: Intel(R) Core(TM) i7-6850K CPU @ 3.60GHz GPU: NVIDIA GTX1080Ti


Note: We thank authors from [1] for the mixup and Pytorch implementation of PreAct ResNet (https://github.com/facebookresearch/mixup-cifar10) \
that we use in our code.

[1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz, "mixup: Beyond Empirical Risk Minimization", in International Conference \
on Learning Representations (ICLR), 2018.