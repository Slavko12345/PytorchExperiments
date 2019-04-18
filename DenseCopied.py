'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F





class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.netName = "Base Net"
    def numParams(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        return pytorch_total_params
    def printNet(self):
        print(self.netName + " will be trained!")
        print("Number of parameters: ", self.numParams())
        
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x




class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, drop = True, pDrop = 0.2):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropFunc = Identity()
        if (drop):
            self.dropFunc = nn.Dropout(pDrop)

    def forward(self, x):
        out = self.dropFunc(self.conv1(F.relu(self.bn1(x))))
        out = self.dropFunc(self.conv2(F.relu(self.bn2(out))))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, drop = True, pDrop = 0.2):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.dropFunc = Identity()
        if (drop):
            self.dropFunc = nn.Dropout(pDrop)

    def forward(self, x):
        out = self.dropFunc(self.conv(F.relu(self.bn(x))))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(BaseNet):
    def __init__(self, block, nblocks, drop = True, pDrop = 0.2, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], drop = drop, pDrop = pDrop)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, drop = drop, pDrop = pDrop)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], drop = drop, pDrop = pDrop)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, drop = drop, pDrop = pDrop)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], drop = drop, pDrop = pDrop)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, drop = drop, pDrop = pDrop)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], drop = drop, pDrop = pDrop)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        
        self.dropFunc = Identity()
        if (drop):
            self.dropFunc = nn.Dropout(pDrop)
            
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, drop = True, pDrop = 0.2):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, drop = drop, pDrop = pDrop))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.dropFunc(out)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)