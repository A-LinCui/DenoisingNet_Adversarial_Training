"""
ResNet with Denoising Blocks in PyTorch on CIAFR10.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Cihang Xie, Yuxin Wu, Laurens van der Maaten, Alan Yuille, Kaiming He
    Feature Denoising for Improving Adversarial Robustness. arXiv:1812.03411

Explanation:
[1] If 'whether_denoising' is True, a ResNet with two denoising blocks will be created.
    In contrast 'whether_denoising' is False, a normal ResNet will be created.
[2] 'filter_type' decides which denoising operation the denoising block will apply.
    Now it includes 'Median_Filter' 'Mean_Filter' and 'Gaussian_Filter'.
[3] 'ksize' means the kernel size of the filter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

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

class denoising_block(nn.Module):
    def __init__(self, in_planes, ksize, filter_type):
        super(denoising_block, self).__init__()
        self.in_planes = in_planes
        self.ksize = ksize
        self.filter_type = filter_type
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.filter_type == 'Median_Filter':
            x_denoised = kornia.median_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Mean_Filter':
            x_denoised = kornia.box_blur(x, (self.ksize, self.ksize))
        elif self.filter_type == 'Gaussian_Filter':
            x_denoised = kornia.gaussian_blur2d(x, (self.ksize, self.ksize), (0.3 * ((x.shape[3] - 1) * 0.5 - 1) + 0.8, 0.3 * ((x.shape[2] - 1) * 0.5 - 1) + 0.8))
        new_x = x + self.conv(x_denoised)
        return new_x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, whether_denoising=False, filter_type="Mean_Filter", ksize=3):
        super(ResNet, self).__init__()
        if whether_denoising:
            self.denoising_block1 = denoising_block(in_planes=64, ksize=ksize, filter_type=filter_type)
            self.denoising_block2 = denoising_block(in_planes=64, ksize=ksize, filter_type=filter_type)
        self.whether_denoising = whether_denoising
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
        if self.whether_denoising:
            out = self.denoising_block1(out)
        out = self.layer1(out)
        if self.whether_denoising:
            out = self.denoising_block2(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(whether_denoising=False, filter_type="Mean_Filter", ksize=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def ResNet34(whether_denoising=False, filter_type="Mean_Filter", ksize=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def ResNet50(whether_denoising=False, filter_type="Mean_Filter", ksize=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def ResNet101(whether_denoising=False, filter_type="Mean_Filter", ksize=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=10, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def ResNet152(whether_denoising=False, filter_type="Mean_Filter", ksize=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=10, whether_denoising=whether_denoising, filter_type=filter_type, ksize=ksize)

def test():
    from torch.autograd import Variable
    net = ResNet18(whether_denoising=True, filter_type="Median_Filter", ksize=3)
    x = Variable(torch.randn(1, 3, 32, 32), requires_grad=True)
    y = net(x)
    print(y)

def denoising_block_test(filter_type):
    from torch.autograd import Variable
    denoising_block1 = denoising_block(in_planes=32, ksize=3, filter_type=filter_type)
    x = Variable(torch.ones(2, 64, 32, 32), requires_grad=True)
    y = denoising_block1(x)
    print(y)
    y.backward(x)
    print(x.grad)

# test()
# denoising_block_test('Median_Filter')
