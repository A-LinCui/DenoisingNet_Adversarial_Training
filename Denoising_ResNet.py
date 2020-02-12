"""
ResNet with Mean Filter Denoising Block in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He
    Feature Denoising for Improving Adversarial Robustness. arXiv:1812.03411

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Mean_Filter(nn.Module):
    def __init__(self, k_size):
        super(Mean_Filter, self).__init__()
        self.k_size = k_size
        if k_size % 2 == 0:
            print('Warning: k_size must be odd!')

    def forward(self, x):
        new_x = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for m in range(x.shape[2]):
                    for n in range(x.shape[3]):
                        total = 0
                        for k in range(max(m - int((self.k_size - 1) / 2), 0), min(m + 1 + int((self.k_size - 1) / 2), x.shape[2])):
                            for g in range(max(n - int((self.k_size - 1) / 2), 0), min(n + 1 + int((self.k_size - 1) / 2), x.shape[3])):
                                total += 1
                                new_x[i][j][m][n] += x[i][j][k][g]
                        new_x[i][j][m][n] /= total
        return new_x


class Median_Filter(nn.Module):
    def __init__(self, k_size):
        super(Median_Filter, self).__init__()
        self.k_size = k_size
        if k_size % 2 == 0:
            print('Warning: k_size must be odd!')

    def forward(self, x):
        new_x = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for m in range(x.shape[2]):
                    for n in range(x.shape[3]):
                        total = 0
                        candidates = torch.zeros((min(m + 1 + int((self.k_size - 1) / 2), x.shape[2]) - max(m - int((self.k_size - 1) / 2), 0)) * (min(n + 1 + int((self.k_size - 1) / 2), x.shape[3]) - max(n - int((self.k_size - 1) / 2), 0)), 1)
                        for k in range(max(m - int((self.k_size - 1) / 2), 0), min(m + 1 + int((self.k_size - 1) / 2), x.shape[2])):
                            for g in range(max(n - int((self.k_size - 1) / 2), 0), min(n + 1 + int((self.k_size - 1) / 2), x.shape[3])):
                                candidates[total] = x[i][j][k][g]
                                total += 1
                        candidates, _ = torch.sort(candidates)
                        new_x[i][j][m][n] = candidates[int((candidates.shape[0] - 1) / 2)]
        return new_x


class denoising_box(nn.Module):
    def __init__(self, in_planes, ksize, filter_type):
        super(denoising_box, self).__init__()
        self.in_planes = in_planes
        if filter_type == 'Mean_Filter':
            self.filter = Mean_Filter(ksize)
        elif filter_type == 'Median_Filter':
            self.filter = Median_Filter(ksize)
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        new_x = x
        x_denoised = self.filter(x)
        x_denoised = self.conv(x_denoised)
        new_x = x + x_denoised
        return new_x


class Denoising_ResNet(nn.Module):
    def __init__(self, filter_type, block, num_blocks, num_classes=10):
        super(Denoising_ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.denoising_box1 = denoising_box(in_planes=64, ksize=3, filter_type=filter_type)
        self.denoising_box2 = denoising_box(in_planes=64, ksize=3, filter_type=filter_type)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.denoising_box1(out)
        out = self.layer1(out)
        out = self.denoising_box2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def Denoising_ResNet18():
    return Denoising_ResNet("Mean_Filter", BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def Denoising_ResNet34():
    return Denoising_ResNet("Mean_Filter", BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def Denoising_ResNet50():
    return Denoising_ResNet("Mean_Filter", Bottleneck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def Denoising_ResNet101():
    return Denoising_ResNet("Mean_Filter", Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

def Denoising_ResNet152():
    return Denoising_ResNet("Mean_Filter", Bottleneck, [3, 8, 36, 3])

def test():
    from torch.autograd import Variable
    net = Denoising_ResNet18()
    x = Variable(torch.randn(1, 3, 32, 32), requires_grad=True)
    y = net(x)
    y.backward(x)

def filter_test():
    from torch.autograd import Variable
    filter = Median_Filter(3)
    x = Variable(torch.randn(1, 3, 32, 32), requires_grad=True)
    y = filter(x)
    print(y)
    y.backward(x)
    print(x.grad)

def denoising_box_test():
    from torch.autograd import Variable
    denoising_box1 = denoising_box(in_planes=32, ksize=3, filter_type='Mean_Filter')
    x = Variable(torch.randn(1, 32, 32, 32), requires_grad=True)
    y = denoising_box1(x)
    print(y)
    y.backward(x)
    print(x.grad)

# test()
# filter_test()
denoising_box_test()