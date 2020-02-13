# Adversarial Training

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import foolbox

import numpy as np
import os
import csv
import time
import argparse
import matplotlib.pyplot as plt

from Denoising_ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

def load_model(basic_model, whether_denoising, filter_type, ksize):
    if basic_model == 'ResNet18':
        model = ResNet18(whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'ResNet34':
        model = ResNet34(whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'ResNet50':
        model = ResNet50(whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'ResNet101':
        model = ResNet101(whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)
    elif basic_model == 'ResNet152':
        model = ResNet152(whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

    model.cuda()
    model = DataParallel(model)

    return model

parser = argparse.ArgumentParser(description='Adversarial Training for Denoising ResNet')
parser.add_argument('-basic_ResNet', dest='basic_model', type=str, default='ResNet18', help='the basic ResNet model including ResNet18/ResNet34/ResNet50/ResNet101')
parser.add_argument('-whether_denoising', dest='whether_denoising', type=bool, default=True, help='whether add denoising block into ResNet')
parser.add_argument('-filter_type', dest='filter_type', type=str, default='Mean_Filter', help='Mean_Filter/Median_Filter/Gaussian_Filter')
parser.add_argument('-kernel_size', dest='ksize', type=int, default=3, help='kernel size of the filter')
parser.add_argument('-epoches', dest='epoches', type=int, default=200, help='epoches')
parser.add_argument('-perturbation_threshold', dest='threshold', type=float, default=1e-8, help='maximum threshold')
parser.add_argument('-batch_size', dest='batch_size', type=int, default=64, help='batch_size')
parser.add_argument('-num_workers', dest='num_workers', type=int, default=32, help='num_workers')
parser.add_argument('-learning_rate', dest='lr', type=float, default=1e-3, help='adv_learning_rate')
parser.add_argument('-weight_decay', dest='weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('-momentum', dest='momentum', type=float, default=0.9, help='momentum')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Load the model
model = load_model(basic_model=args.basic_model, whether_denoising=args.whether_denoising, filter_type=args.filter_type, ksize=args.ksize)

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
GPU_COUNT = torch.cuda.device_count()
testset = torchvision.datasets.CIFAR10(root='/home/eva_share/datasets/cifar10', train=False, download=True, transform=transform_test)
trainset = torchvision.datasets.CIFAR10(root='/home/eva_share/datasets/cifar10', train=True, download=True, transform=transform_train)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size * GPU_COUNT, shuffle=False, num_workers=args.num_workers)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size * GPU_COUNT, shuffle=False, num_workers=args.num_workers)

# Establish the training log
with open(args.basic_model + 'adv_training_log.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "test_clean_acc", "test_adv_acc", "train_clean_loss", "train_adv_loss", "train_loss", "train_clean_acc", "train_adv_acc"])

# Training settings
criterion = nn.CrossEntropyLoss()
params = list(model.parameters())
optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
scheduler = MultiStepLR(optimizer, milestones=[int(epoch/2), int(epoch*3/4), int(epoch*7/8)], gamma=0.1)

# Record the best accuracy
best_test_clean_acc, best_test_adv_acc, best_epoch = 0, 0, 0

for epoch in range(args.epoches):

    print("basic model: {}, whether denoising: {}, filter type: {}, kernel size: {}, max allowed perturbation: {}".format(args.basic_model, args.whether_denoising, args.filter_type, args.ksize, args.threshold))

    scheduler.step()

    # Test on the testset
    model.eval()
    test_total, test_correct, test_robustness = 0, 0, 0

    # Setup the adversary
    fmodel = foolbox.models.PyTorchModel(model, bounds=(-255, 255), num_classes=10, device='Parrallel')
    attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.MeanSquaredDistance)

    start_time = time.time()

    for (images, labels) in testloader:
        images = images.numpy()
        labels = labels.numpy()
        test_total += images.shape[0]
        test_correct += np.sum(fmodel.forward(images).argmax(axis=-1) == labels)
        adversarials = attack(images, labels, unpack=False)
        for adversarial in adversarials:
            if adversarial.perturbed is None or adversarial.distance.value > args.threshold:
                test_robustness += 1

    test_acc, test_adv_acc = test_correct / test_total, test_robustness / test_total

    # Record the time on the testset
    end_time = time.time()
    testset_total_time = end_time - start_time

    # Save the present best model and its statistics
    if test_adv_acc > best_test_adv_acc:
        best_epoch = epoch
        best_test_adv_acc = test_adv_acc
        best_test_clean_acc = test_acc
        torch.save(model.state_dict(), args.basic_model + '.pkl')

    print("Present best adversarial model ----- best epoch: {} clean_test_acc: {:.3f} adv_test_acc: {:.3f}".format(best_epoch, best_test_clean_acc, best_test_adv_acc))
    print("epoch:{} Classification accuracy on clean testset: {:.3f}".format(epoch, test_acc))
    print("epoch:{} Classification accuracy on adversarial testset: {:.3f}".format(epoch, test_adv_acc))

    # Test and Train on the trainset
    train_total, train_correct, train_robustness = 0, 0, 0
    train_clean_loss, train_adv_loss, train_loss = 0, 0, 0

    start_time = time.time()

    for (images, labels) in trainloader:
        fmodel = foolbox.models.PyTorchModel(model, bounds=(-255, 255), num_classes=10, device='Parrallel')
        attack = foolbox.attacks.FGSM(fmodel, distance=foolbox.distances.MeanSquaredDistance)
        images = images.numpy()
        labels = labels.numpy()
        train_total += images.shape[0]
        train_correct += np.sum(fmodel.forward(images).argmax(axis=-1) == labels)

        adversarials = attack(images, labels, unpack=False)
        adversarial_images, adversarial_orig_labels = [], []

        for adversarial in adversarials:
            if adversarial.perturbed is None or adversarial.distance.value > args.threshold:
                train_robustness += 1
            else:
                adversarial_images.append(adversarial.perturbed)
                adversarial_orig_labels.append(adversarial.original_class)

        adversarial_images = torch.Tensor(adversarial_images).cuda()
        adversarial_orig_labels = torch.Tensor(adversarial_orig_labels).cuda()

        # Adversarial Training Both on Adversarial Examples and Clean Inputs
        model.train()
        adv_outputs = model(adversarial_images)
        adv_loss = criterion(adv_outputs, adversarial_orig_labels.long())
        clean_outputs = model(torch.from_numpy(images).cuda())
        clean_loss = criterion(clean_outputs, torch.from_numpy(labels).cuda())
        optimizer.zero_grad()
        loss = clean_loss + adv_loss
        loss.backward()
        optimizer.step()
        train_clean_loss += clean_loss.item()
        train_adv_loss += adv_loss.item()
        train_loss = train_adv_loss + train_clean_loss
        model.eval()

    # Record the time on the trainset
    end_time = time.time()
    trainset_total_time = end_time - start_time

    train_acc, train_adv_acc = train_correct / train_total, train_robustness / train_total
    print("epoch:{} train_clean_loss: {:.3f} train_adv_loss: {:.3f} train_total_loss: {:.3f}".format(epoch, train_clean_loss, train_adv_loss, train_loss))
    print("epoch:{} Classification accuracy on clean trainset: {:.3f}".format(epoch, train_acc))
    print("epoch:{} Classification accuracy on adversarial trainset: {:.3f}\n".format(epoch,  train_adv_acc))
    print("epoch:{} Consumed time on the testset: {:.5f}s  on the trainset: {:.5f}s".format(epoch, testset_total_time, trainset_total_time))

    # Recond the statistics
    with open(args.basic_model + 'adv_training_log.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, test_acc, test_adv_acc, train_clean_loss, train_adv_loss, train_loss, train_acc, train_adv_acc])

    # Load the statistics in the training log
    epoch, test_acc, testset_adv_acc, train_clean_loss, train_adv_loss, train_loss, train_acc, train_adv_acc = [], [], [], [], [], [], [], []
    with open(args.basic_model + 'adv_training_log.csv', 'a') as f:
        reader = csv.reader(f)
        for i in reader:
            if i > 0:
                epoch.append(i[0])
                test_acc.append(i[1])
                test_adv_acc.append(i[2])
                train_clean_loss.append(i[3])
                train_adv_loss.append(i[4])
                train_loss.append(i[5])
                train_acc.append(i[6])
                train_adv_acc.append(i[7])

    # Generate the training accuracy line
    plt.figure()
    plt.plot(epoch, test_acc, label='test_accuracy', linewidth=2, color='r')
    plt.plot(epoch, test_adv_acc, label='test_adv_accuracy', linewidth=2, color='b')
    plt.plot(epoch, train_acc, label='train_accuracy', linewidth=2, color='y')
    plt.plot(epoch, train_adv_acc, label='train_adv_accuracy', linewidth=2, color='g')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.xlim(-1, max(epoch) + 1)
    plt.ylim(0, 1.0)
    plt.title("training accuracy")
    plt.savefig("train_accuracy.png")

    # Generate the training loss line
    plt.figure()
    plt.plot(epoch, train_loss, label='train_total_loss', linewidth=2, color='r')
    plt.plot(epoch, train_clean_loss, label='train_clean_loss', linewidth=2, color='b')
    plt.plot(epoch, train_adv_loss, label='train_adv_loss', linewidth=2, color='y')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xlim(-1, max(epoch) + 1)
    plt.ylim(0, max(train_loss))
    plt.title("training loss on the trainset")
    plt.savefig("train_loss.png")