# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import PreResNet
import math
import torchvision.models as models
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from utils import *

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='This is the official implementation for the ICML 2019 Unsupervised label noise modeling and loss correction paper. This work is under MIT licence. Please refer to the RunScripts.sh and README.md files for example usages. Consider citing our work if this code is usefull for your project')
    parser.add_argument('--root-dir', type=str, default='.', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training, default: 128')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='dataset to train on, default: CIFAR10')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default: 0.9')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA support')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed, set it to go to determinist mode. We used 1 for the paper, default: None')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status, default: 10')
    parser.add_argument('--noise-level', type=float, default=80.0,
                        help='percentage of noise added to the data (values from 0. to 100.), default: 80.')
    parser.add_argument('--experiment-name', type=str, default='runs',
                        help='name of the experiment for the output files storage, default: runs')
    parser.add_argument('--alpha', type=float, default=32, help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--M', nargs='+', type=int, default=[100, 250],
                        help="Milestones for the LR sheduler, default 100 250")
    parser.add_argument('--Mixup', type=str, default='None', choices=['None', 'Static', 'Dynamic'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--reg-term', type=float, default=0., 
                        help="Parameter of the regularization term, default: 0.")


    args = parser.parse_args()
    
    print(args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed:
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(args.seed)  # CPU seed
        if device == "cuda":
            torch.cuda.manual_seed_all(args.seed)  # GPU seed

        random.seed(args.seed)  # python seed for image transformation

    # CIFAR meta
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    if args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR10(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        trainset = datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR100(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 100
    else:
        raise NotImplementedError
        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    train_loader_track = torch.utils.data.DataLoader(trainset_track, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    model = PreResNet.ResNet18(num_classes=num_classes).to(device)

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    labels = get_data_cifar_2(train_loader_track)  # it should be "clonning"
    noisy_labels = add_noise_cifar_w(train_loader, args.noise_level)  # it changes the labels in the train loader directly
    noisy_labels_track = add_noise_cifar_w(train_loader_track, args.noise_level)

    # path where experiments are saved
    exp_path = os.path.join('./', 'noise_models_PreResNet18_{0}'.format(args.experiment_name), str(args.noise_level))

    if not os.path.isdir(exp_path):
         os.makedirs(exp_path)

    bmm_model=bmm_model_maxLoss=bmm_model_minLoss=cont=k = 0

    bootstrap_ep_std = milestones[0] + 5 + 1 # the +1 is because the conditions are defined as ">" or "<" not ">="
    guidedMixup_ep = 106

    if args.Mixup == 'Dynamic':
        bootstrap_ep_mixup = guidedMixup_ep + 5
    else:
        bootstrap_ep_mixup = milestones[0] + 5 + 1

    countTemp = 1

    temp_length = 200 - bootstrap_ep_mixup


    for epoch in range(1, args.epochs + 1):
        # train
        scheduler.step()

        ### Standard CE training (without mixup) ###
        if args.Mixup == "None":
            print('\t##### Doing standard training with cross-entropy loss #####')
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch)

        ### Mixup ###
        if args.Mixup == "Static":
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, 32)

            else:
                if args.BootBeta == "Hard":
                    print("\t##### Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,\
                                                                                    alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)
                elif args.BootBeta == "Soft":
                    print("\t##### Doing SOFT BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, \
                                                                                    alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes)

        ## Dynamic Mixup ##
        if args.Mixup == "Dynamic":
            alpha = args.alpha
            if epoch < guidedMixup_ep:
                print('\t##### Doing NORMAL mixup for {0} epochs #####'.format(guidedMixup_ep - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, 32)

            elif epoch < bootstrap_ep_mixup:
                print('\t##### Doing Dynamic mixup from epoch {0} #####'.format(guidedMixup_ep))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,\
                                                                        bmm_model_maxLoss, bmm_model_minLoss)
            else:
                print("\t##### Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                loss_per_epoch, acc_train_per_epoch_i, countTemp, k = train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, \
                                                                                                                epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, \
                                                                                                                countTemp, k, temp_length, args.reg_term, num_classes)
        ### Training tracking loss
        epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
            track_training_loss(args, model, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)

        # test
        loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args, model, device, test_loader)

        if epoch == 1:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]

                if cont>0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont+=1

        if epoch == args.epochs:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestValLoss_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))


if __name__ == '__main__':
    main()
