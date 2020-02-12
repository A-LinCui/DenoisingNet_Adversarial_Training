# Adversarial Training

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import foolbox

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from Denoising_ResNet import Denoising_ResNet18

parser = argparse.ArgumentParser(description='Adversarial Attack')
parser.add_argument('-perturbation_threshold', dest='threshold', type=float, default=1e-8, help='maximum threshold')
parser.add_argument('-epoches', dest='epoches', type=int, default=200, help='epoches')
parser.add_argument('-batch_size', dest='batch_size', type=int, default=64, help='batch_size')
parser.add_argument('-num_workers', dest='num_workers', type=int, default=32, help='num_workers')
parser.add_argument('-learning_rate', dest='lr', type=float, default=1e-4, help='adv_learning_rate')
parser.add_argument('-weight_decay', dest='weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('-momentum', dest='momentum', type=float, default=0.9, help='momentum')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Load the model
model = Denoising_ResNet18()
model.cuda()
model = DataParallel(model)

GPU_COUNT = torch.cuda.device_count()

# Setup the transformation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load the dataset
testset = torchvision.datasets.CIFAR10(root='/home/eva_share/datasets/cifar10', train=False, download=True, transform=transform_test)
trainset = torchvision.datasets.CIFAR10(root='/home/eva_share/datasets/cifar10', train=True, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size*GPU_COUNT, shuffle=False, num_workers=args.num_workers)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size*GPU_COUNT, shuffle=False, num_workers=args.num_workers)

for epoch in range(args.epoches):

    # Test
    model.eval()
    total, correct, robustness = 0, 0, 0

    # Setup the adversary
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-255, 255), num_classes=10, device='Parrallel')
    attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.MeanSquaredDistance)

    for (images, labels) in testloader:
        images = images.numpy()
        labels = labels.numpy()
        total += images.shape[0]
        correct += np.sum(fmodel.forward(images).argmax(axis=-1) == labels)
        adversarials = attack(images, labels, unpack=False)
        for adversarial in adversarials:
            if adversarial.perturbed is None or adversarial.distance.value > args.threshold:
                robustness += 1

    print("epoch:{} Classification accuracy on clean testset: {:.3f}".format(epoch, correct / total))
    print("epoch:{} Classification accuracy on adversarial testset: {:.3f}".format(epoch, robustness / total))

    # Train
    total, correct, robustness = 0, 0, 0
    for (images, labels) in trainloader:
        fmodel = foolbox.models.PyTorchModel(model, bounds=(-255, 255), num_classes=10, device='Parrallel')
        attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.MeanSquaredDistance)
        images = images.numpy()
        labels = labels.numpy()
        total += images.shape[0]
        correct += np.sum(fmodel.forward(images).argmax(axis=-1) == labels)

        adversarials = attack(images, labels, unpack=False)
        adversarial_images, adversarial_orig_labels = [], []

        for adversarial in adversarials:
            if adversarial.perturbed is None or adversarial.distance.value > args.threshold:
                robustness += 1
            else:
                adversarial_images.append(adversarial.perturbed)
                adversarial_orig_labels.append(adversarial.original_class)

        adversarial_images = torch.Tensor(adversarial_images).cuda()
        adversarial_orig_labels = torch.Tensor(adversarial_orig_labels).cuda()

        # Adversarial Training Both on Adversarial Examples and Clean Inputs
        model.train()
        criterion = nn.CrossEntropyLoss()
        params = list(model.parameters())
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
        adv_outputs = model(adversarial_images)
        adv_loss = criterion(adv_outputs, adversarial_orig_labels.long())
        clean_outputs = model(torch.from_numpy(images).cuda())
        clean_loss = criterion(clean_outputs, torch.from_numpy(labels).cuda())
        optimizer.zero_grad()
        loss = clean_loss + adv_loss
        loss.backward()
        optimizer.step()
        model.eval()

    print("epoch:{} Classification accuracy on clean trainset: {:.3f}".format(epoch, correct / total))
    print("epoch:{} Classification accuracy on adversarial trainset: {:.3f}".format(epoch,  robustness / total))
    print()