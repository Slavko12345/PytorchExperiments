#DenseReluNet 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

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

        
class NearestNeighborNet(BaseNet):
    def __init__(self, numBlocks, numNeighbors):
        super().__init__()
        self.numBlocks = numBlocks
        self.numNeighbors = numNeighbors
        self.neighbors = torch.nn.Parameter((torch.rand(numBlocks, numNeighbors,3, 32, 32)-0.5)*1)
        self.fc = nn.Linear(numBlocks, 10)
    
    def forward(self, x):
        dist = (torch.abs(x.unsqueeze(1).unsqueeze(1) - self.neighbors)).view(x.shape[0], self.numBlocks, self.numNeighbors, -1)\
        .max(dim = 3)[0].min(dim = 2)[0]*(-1.0)
        return self.fc(dist)
    

def expandConv(x, cuda_av, rows = 3, cols = 3):
    dev = None
    if (cuda_av):
        dev = "cuda:0"
    y = torch.zeros(x.shape[0], x.shape[1], rows * cols, x.shape[2], x.shape[3], device = dev)
    ind = 0
    rowBound = int((rows - 1)/2)
    colBound = int((cols - 1)/2)
    for row in range(-rowBound, rowBound + 1):
        for col in range(-colBound, colBound+1):
            y[:, :, ind,\
            max(row, 0): min(x.shape[2]+row, x.shape[2]), max(col, 0): min(x.shape[3]+col, x.shape[3])] \
            =x[:,:,max(-row, 0): min(x.shape[2]-row, x.shape[2]), max(-col, 0): min(x.shape[3]-col, x.shape[3])]
            ind += 1
    return y  
        
        
class ConvNearestNeightbor(nn.Module):
    def __init__(self, num, depth, rows = 3, cols = 3):
        super().__init__()
        self.num = num
        self.rows = rows
        self.cols = cols
        self.neighbors = torch.nn.Parameter((torch.rand(num, depth, 9)-0.5)*1)
        self.cuda_av = torch.cuda.is_available()
    
    def forward(self, x):
        y = expandConv(x, self.cuda_av)
        diff = y.unsqueeze(1) - self.neighbors.unsqueeze(0).unsqueeze(4).unsqueeze(5)
        return torch.abs(diff).max(dim = 3)[0].reshape(x.shape[0], self.num * x.shape[1], x.shape[2], x.shape[3])
    

class ConvNNN(BaseNet):
    def __init__(self, numBlocks, numFeatures):
        super().__init__()
        self.numFeatures = numFeatures
        self.numBlocks = numBlocks
        self.convNNList = nn.ModuleList()
        depth = 3
        for i in range(numBlocks):
            self.convNNList.append(ConvNearestNeightbor(numFeatures, depth))
            depth = depth * numFeatures
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(depth, 10)
        
    def forward(self, x):
        for i in range(self.numBlocks):
            x = self.convNNList[i](x)
        x = -x
        x = self.pool(x)
        x.squeeze_()
        return self.fc(x)
    
    
    
    
class QuadraticNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Quadratic Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.convList[i](x)
            y = F.relu(y)
            x = torch.cat((x, y), dim = 1)
            print(x.shape)
        x = self.pool(-x).squeeze()
        x = self.fc(x)
        return x 
    
    
        
        
class _KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        batch_size, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze_(dim=1), active_indices] = 1
        tensor[~mask_active] = 0
        tensor[mask_active] = 1
        # ctx.save_for_backward(mask_active)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class KWinnersTakeAll(nn.Module):

    def __init__(self, sparsity=0.05):
        super().__init__()
        assert 0. <= sparsity <= 1., "Sparsity should lie in (0, 1) interval"
        self.sparsity = sparsity

    def forward(self, x):
        x = _KWinnersTakeAllFunction.apply(x, self.sparsity)
        return x

    def extra_repr(self):
        return f'sparsity={self.sparsity}'


class KWinnersTakeAllSoft(KWinnersTakeAll):

    def __init__(self, sparsity=0.05, hardness=10):
        super().__init__(sparsity)
        self.hardness = hardness

    def forward(self, x):
        if self.training:
            batch_size, embedding_size = x.shape
            _, argsort = x.sort(dim=1, descending=True)
            k_active = math.ceil(self.sparsity * embedding_size)
            range_idx = torch.arange(batch_size)
            kth_element = x[range_idx, argsort[:, k_active]]
            if k_active < embedding_size:
                kth_next = x[range_idx, argsort[:, k_active+1]]
                threshold = (kth_element + kth_next) / 2
            else:
                threshold = kth_element
            threshold.unsqueeze_(dim=1)
            x_scaled = self.hardness * (x - threshold)
            return x_scaled.sigmoid()
        else:
            return _KWinnersTakeAllFunction.apply(x, self.sparsity)

    def extra_repr(self):
        old_repr = super().extra_repr()
        return f"{old_repr}, hardness={self.hardness}"
        
        
        
class LogicLayer(nn.Module):
    def __init__(self, inputDepth, layerSize):
        super().__init__()
        self.inputDepth = inputDepth
        self.layerSize = layerSize
        self.convParams = Parameter(torch.Tensor(2 * layerSize, inputDepth, 3, 3))
        self.convParams = torch.nn.Parameter((torch.rand_like(self.convParams)-0.5)*0.05)
        self.convBias = Parameter(torch.Tensor(2 * layerSize))
        self.convBias = torch.nn.Parameter((torch.rand_like(self.convBias)-0.5)*0.05)
    
    def forward(self, x):
        positiveParams = F.relu(self.convParams)
        positiveBias = F.relu(self.convBias)
        out = F.conv2d(x, positiveParams, bias = positiveBias, padding = 1)
        return torch.cat((out, out.view(x.shape[0], self.layerSize, 2, x.shape[2], x.shape[3]).min(dim=2)[0]), dim = 1)    
    

    
    
class LogicNet(BaseNet):
    def __init__(self, initDepth, nLayers, layerSize):
        super().__init__()
        self.netName = "Logic Net"
        self.initDepth = initDepth
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.logicList1 = nn.ModuleList()
        self.logicList2 = nn.ModuleList()
        self.logicList3 = nn.ModuleList()
        
        self.initConv = nn.Conv2d(3, self.initDepth, 3, padding = 1)
        
        depth = self.initDepth
        for i in range(self.nLayers):
            self.logicList1.append(LogicLayer(depth, self.layerSize))
            depth += self.layerSize * 3
            
        for i in range(self.nLayers):
            self.logicList2.append(LogicLayer(depth, self.layerSize))
            depth += self.layerSize * 3
            
        for i in range(self.nLayers):
            self.logicList3.append(LogicLayer(depth, self.layerSize))
            depth += self.layerSize * 3
            
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(8, 8)        
        
        self.fc = nn.Linear(depth, 10)
        
    def forward(self, x):
        x = F.relu(self.initConv(x))
        
        for i in range(self.nLayers):
            x = torch.cat((x, self.logicList1[i](x)), dim = 1)
        x = self.pool1(x)
        
        for i in range(self.nLayers):
            x = torch.cat((x, self.logicList2[i](x)), dim = 1)
        x = self.pool2(x)
        
        for i in range(self.nLayers):
            x = torch.cat((x, self.logicList3[i](x)), dim = 1)
        x = self.pool3(x).squeeze()
        
        x = self.fc(x)
        return x

    

class SimplifiedLogicLayer(nn.Module):
    def __init__(self, inputDepth, layerSize):
        super().__init__()
        self.inputDepth = inputDepth
        self.layerSize = layerSize
        self.conv = nn.Conv2d(inputDepth, 2 * layerSize, 3, padding = 1)
    
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = out.view(x.shape[0], self.layerSize, 2, x.shape[2], x.shape[3])
        return out.min(dim = 2)[0]
        #return torch.cat((x, x.min(dim=1, keepdim = True)[0]), dim = 1)
    
    
class SimplifiedLogicNet(BaseNet):
    def __init__(self, nLayers, layerSize):
        super().__init__()
        self.netName = "Simplified Logic Net"
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.logicList1 = nn.ModuleList()
        self.logicList2 = nn.ModuleList()
        self.logicList3 = nn.ModuleList()
                
        depth = 3
        for i in range(self.nLayers):
            self.logicList1.append(SimplifiedLogicLayer(depth, self.layerSize))
            depth += self.layerSize
            
        for i in range(self.nLayers):
            self.logicList2.append(SimplifiedLogicLayer(depth, self.layerSize))
            depth += self.layerSize
            
        for i in range(self.nLayers):
            self.logicList3.append(SimplifiedLogicLayer(depth, self.layerSize))
            depth += self.layerSize
            
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(8, 8)        
        
        self.fc = nn.Linear(depth, 10)
        
    def forward(self, x):
        for i in range(self.nLayers):
            x = torch.cat((x, self.logicList1[i](x)), dim = 1)
        x = self.pool1(x)
        
        for i in range(self.nLayers):
            x = torch.cat((x, self.logicList2[i](x)), dim = 1)
        x = self.pool2(x)
        
        for i in range(self.nLayers):
            x = torch.cat((x, self.logicList3[i](x)), dim = 1)
        x = self.pool3(x).squeeze()
        
        x = self.fc(x)
        return x
    
    
    
    
    
def normNonLin(input):
    return torch.exp(-torch.abs(input))



class DenseNormNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dense Relu Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            x = torch.cat((x, normNonLin(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, normNonLin(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, normNonLin(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.view(-1, 3 + 3 * self.numBlocks * self.blockSize)
        x = self.fc(x)
        return x
    
    
    

class DenseReluNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dense Relu Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.view(-1, 3 + 3 * self.numBlocks * self.blockSize)
        x = self.fc(x)
        return x
    
    
    
    
class BinaryUnit(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, a = 1):
        if (self.training):
            return torch.sigmoid(a * x)
        else:
            return (torch.sign(x) + 1) * 0.5
    
    
    
    
class DenseBinaryNet(BaseNet):
    def __init__(self, initialDepth, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dense Binary Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.binary = BinaryUnit()
        
        self.initialConv = nn.Conv2d(3, initialDepth, 3, padding = 1)
        
        size = initialDepth
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x, alpha = 1):
        x = self.initialConv(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, self.binary(self.convList1[i](x), alpha)), dim = 1)
        x = self.pool1(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, self.binary(self.convList2[i](x), alpha)), dim = 1)
        x = self.pool2(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, self.binary(self.convList3[i](x), alpha)), dim = 1)
        x = self.pool3(x)
        
        x = x.squeeze()
        x = self.fc(x)
        return x    
    
    
    
    
    
    
    
    
    

class LinearInvariantConv(nn.Module):
    def __init__(self, inputDepth, layerSize, invarianceLevel = 0):
        #0: conv with bias
        #1: conv no bias
        #2: sum zero [out, :, :, :]
        #3: sum zero [out, in, :, :]
        #4: sum zero [out, in, row, :] and [out, in, :, col]
        
        super().__init__()
        self.invarianceLevel = invarianceLevel
        
        self.biasParams = None
        
        if (invarianceLevel >= 0 and invarianceLevel <= 3):
            self.convParams = Parameter(torch.Tensor(layerSize, inputDepth, 3, 3))
            if (invarianceLevel == 0):
                self.biasParams = Parameter(torch.Tensor(layerSize))
                nn.init.uniform_(self.biasParams, -0.1, 0.1)
            self.sh = self.convParams.shape
        
        if (invarianceLevel == 4):
            self.convParams = Parameter(torch.Tensor(layerSize, inputDepth, 4))
            self.Basis = torch.Tensor([[[1, -1, 0], [-1, 1, 0], [0, 0, 0]], 
                                                 [[0, -1, 1], [0, 1, -1], [0, 0, 0]], 
                                                 [[0, 0, 0], [-1, 1, 0], [1, -1, 0]],
                                                 [[0, 0, 0], [0, 1, -1], [0, -1, 1]]])
            if torch.cuda.is_available():
                self.Basis = self.Basis.cuda()
        
        nn.init.uniform_(self.convParams, -0.1, 0.1)
        
    def forward(self, x):   
        kernel = self.convParams
        if (self.invarianceLevel == 2 or self.invarianceLevel == 3):
            if (self.invarianceLevel == 2):
                reduct = self.convParams.view(self.sh[0], -1).mean(dim=1).view(-1,1,1,1)
            else:
                reduct = self.convParams.view(self.sh[0], self.sh[1], -1).mean(dim=2).view(self.sh[0],self.sh[1],1,1)
            kernel = self.convParams - reduct
        if (self.invarianceLevel == 4):
            kernel = torch.einsum("ijk,kmn->ijmn", (self.convParams, Variable(self.Basis, requires_grad = False)))
        
        return F.conv2d(x, kernel, bias = self.biasParams, padding = 1)    
    
    
    
class LinInvConvMaxMin(nn.Module):
    def __init__(self, inputDepth, layerSize, invarianceLevel = 0):
        super().__init__()
        self.conv = LinearInvariantConv(inputDepth, layerSize, invarianceLevel = invarianceLevel)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], int(x.shape[1] / 2), 2, x.shape[2], x.shape[3])
        return torch.cat((x.min(dim = 2)[0], x.max(dim = 2)[0]), dim = 1)
        
    
class DenseLinearlyInvariantNet(BaseNet):
    def __init__(self, initialDepth, numBlocks, blockSize, inputInvLevel = 0, innerInvLevel = 0):
        super().__init__()
        self.netName = "Dense Linearly Invariant Net"
        self.numBlocks = numBlocks
        
        self.initialConv = LinearInvariantConv(3, initialDepth, inputInvLevel)
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = initialDepth
        for i in range(numBlocks):
            self.convList1.append(LinInvConvMaxMin(size, blockSize, innerInvLevel))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(LinInvConvMaxMin(size, blockSize, innerInvLevel))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(LinInvConvMaxMin(size, blockSize, innerInvLevel))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        x = self.initialConv(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, self.convList1[i](x)), dim = 1)
        x = self.pool1(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, self.convList2[i](x)), dim = 1)
        x = self.pool2(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, self.convList3[i](x)), dim = 1)
        x = self.pool3(x).squeeze()
        
        x = self.fc(x)
        return x    
    
 
    
    
    
    
    

        
#DenseBNReluNet 
class DenseBNReluNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dense BN Relu Net"
        
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.BNList1 = nn.ModuleList()
        self.BNList2 = nn.ModuleList()
        self.BNList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            self.BNList1.append(nn.BatchNorm2d(num_features = blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            self.BNList2.append(nn.BatchNorm2d(num_features = blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            self.BNList3.append(nn.BatchNorm2d(num_features = blockSize))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.BNList1[i](self.convList1[i](x)))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.BNList2[i](self.convList2[i](x)))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.BNList3[i](self.convList3[i](x)))), dim = 1)
        x = self.pool3(x)
        x = x.view(-1, 3 + 3 * self.numBlocks * self.blockSize)
        x = self.fc(x)
        return x
        
        
        
        
        
#DenseMaxMinNet 
def maxmin_fc(input):
    max_t = input.clamp(min=0)
    min_t = input.clamp(max=0)
    return torch.cat((max_t, min_t), dim=1)


def MaxMinPairwise(input):
    res = input.view(input.shape[0], int(input.shape[1] / 2), 2, input.shape[2], input.shape[3])
    return torch.cat((res.max(dim = 2)[0], res.min(dim = 2)[0]), dim = 1)



class DenseMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dense Max Min Net"
        
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, int(blockSize/2), 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, int(blockSize/2), 3, padding = 1))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, int(blockSize/2), 3, padding = 1))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            x = torch.cat((x, maxmin_fc(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, maxmin_fc(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, maxmin_fc(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.view(-1, 3 + 3 * self.numBlocks * self.blockSize)
        x = self.fc(x)
        return x
    
    
    
    
    
#DenseBNMaxMinNet 
class DenseBNMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dense BN Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.BNList1 = nn.ModuleList()
        self.BNList2 = nn.ModuleList()
        self.BNList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, int(blockSize/2), 3, padding = 1))
            self.BNList1.append(nn.BatchNorm2d(num_features = int(blockSize/2)))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, int(blockSize/2), 3, padding = 1))
            self.BNList2.append(nn.BatchNorm2d(num_features = int(blockSize/2)))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, int(blockSize/2), 3, padding = 1))
            self.BNList3.append(nn.BatchNorm2d(num_features = int(blockSize/2)))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            x = torch.cat((x, maxmin_fc(self.BNList1[i](self.convList1[i](x)))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, maxmin_fc(self.BNList2[i](self.convList2[i](x)))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, maxmin_fc(self.BNList3[i](self.convList3[i](x)))), dim = 1)
        x = self.pool3(x)
        x = x.view(-1, 3 + 3 * self.numBlocks * self.blockSize)
        x = self.fc(x)
        return x
    
    
class ColumnDrop(nn.Module):
    def __init__(self, pDrop, pNotDrop):
        super().__init__()
        self.pDrop = pDrop
        self.pNotDrop = pNotDrop
        
    def forward(self, x):
        if (not self.training):
            return x
        #if (torch.bernoulli(torch.tensor([self.pNotDrop])).item()):
        #    return x
        active = x.new(x.shape[0], 1, x.shape[2], x.shape[3])
        active.bernoulli_(1.0 - self.pDrop).mul_(1.0/(1.0 - self.pDrop))
        return x.mul(active) 
    
    
class ColumnCrop(nn.Module):
    def __init__(self, pRemain):
        super().__init__()
        self.pRemain = pRemain
        
    def forward(self, x):
        if (not self.training):
            return x
        mb = x.shape[0]
        dth = x.shape[1]
        rws = x.shape[2]
        cls = x.shape[3]
        
        sizeRow = int(rws * self.pRemain)
        sizeCol = int(cls * self.pRemain)
        
        startRow = torch.randint(low=0, high=rws - sizeRow + 1, size=(mb,1), dtype=torch.long, device = x.device).view(-1, 1).float()
        startCol = torch.randint(low=0, high=cls - sizeCol + 1, size=(mb,1), dtype=torch.long, device = x.device).view(-1, 1).float()
        
        indRow = torch.arange(0, rws, device=x.device).view(1, -1).float()
        indCol = torch.arange(0, cls, device=x.device).view(1, -1).float()
        
        upperCondition = (indRow >= startRow).view(mb, rws, 1) * (indCol >= startCol).view(mb, 1, cls)
        lowerCondition = (indRow < startRow+sizeRow).view(mb, rws, 1) * (indCol < startCol+sizeCol).view(mb, 1, cls)
        mask = upperCondition * lowerCondition
        multiplier = (rws * cls) / (sizeRow * sizeCol)
        
        return x.mul(mask.view(mb, 1, rws, cls).float()).mul(multiplier)
    
    
    
    

    
class FullColumnCropPool(nn.Module):
    def __init__(self, pRemain):
        super().__init__()
        self.pRemain = pRemain
        self.crop = ColumnCrop(pRemain)
    
    def forward(self, x):
        y=self.crop(x)
        return y.view(x.shape[0], x.shape[1], -1).mean(dim=2)
    
    
    
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
    
class LinearTransform(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.scale = Parameter(torch.Tensor(num_features))
        self.bias  = Parameter(torch.Tensor(num_features))
        nn.init.uniform_(self.scale, -1, 1)
        nn.init.uniform_(self.bias, -1, 1)
    
    def forward(self, x):
        return x * self.scale.view(1, self.num_features, 1, 1) + self.bias.view(1, self.num_features, 1, 1)
        
        
class DenseBlock(nn.Module):
    def __init__(self, initSize, numBlocks, blockSize, bottleneckSize, 
                 dropRegime = "None", pDrop = 0, bottleDrop = True, convDrop = False, 
                 bn = True, nonlin = 'Relu', bottleBn = True, bottleNonlin = True, bnBeforeConv = True,
                 bnRegime = "bn", bias = False):
        super().__init__()
        self.initSize = initSize
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.bottleDrop = bottleDrop
        self.convDrop = convDrop
        self.bn = bn
        self.bias = bias
        self.nonlin = nonlin
        self.pDrop = pDrop
        self.bottleBn = bottleBn
        self.bottleNonlin = bottleNonlin
        self.bnBeforeConv = bnBeforeConv
        
        assert dropRegime in ["None", "Drop", "ColumnDrop"]
        assert nonlin in ["Identity", "Relu", "MaxMin", "MaxMinPairwise"]
        assert bnRegime in ["bn", "ln"]
        
        if (bnRegime == "bn"):
            bnModule = nn.BatchNorm2d
        else:
            bnModule = LinearTransform
        
        self.vertConvList = nn.ModuleList()
        self.convList = nn.ModuleList()
        if (self.bn):
            self.bnList = nn.ModuleList()
        
        if (self.bottleBn):
            self.bottleBnList = nn.ModuleList()
            
        if (bnBeforeConv):
            self.convBnList1 = nn.ModuleList()
            self.convBnList2 = nn.ModuleList()
        
        self.drop = Identity()
        if (dropRegime == "Drop"):
            self.drop = nn.Dropout(self.pDrop)
        if (dropRegime == "ColumnDrop"):
            self.drop = ColumnDrop(self.pDrop, 0)
        
        if (self.nonlin == "Identity"):
            self.activation = Identity()
        if (self.nonlin == "Relu"):
            self.activation = F.relu
        if (self.nonlin == "MaxMin"):
            self.activation = maxmin_fc
        if (self.nonlin == "MaxMinPairwise"):
            self.activation = MaxMinPairwise
        
        if (nonlin == "MaxMin"):
            mainSizeInc = 2 * blockSize
        else:
            mainSizeInc = blockSize
            
        convSizeInp = bottleneckSize
        if (bottleNonlin and nonlin == "MaxMin"):
            convSizeInp = 2 * bottleneckSize
        
        size = self.initSize
        for i in range(numBlocks):
            self.vertConvList.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            if (self.bottleBn):
                self.bottleBnList.append(bnModule(num_features = bottleneckSize))
            if (self.bnBeforeConv):
                self.convBnList1.append(bnModule(num_features = convSizeInp))
            self.convList.append(nn.Conv2d(convSizeInp, blockSize, 3, padding = 1, bias = self.bias))
            if (self.bn):
                self.bnList.append(bnModule(num_features = blockSize))
            if (self.bnBeforeConv):
                self.convBnList2.append(bnModule(num_features = mainSizeInc))
            size += mainSizeInc
        
        self.finalSize = size
            
    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList[i](x)
            if (self.bottleBn):
                y = self.bottleBnList[i](y)
            if (self.bottleNonlin):
                y = self.activation(y)
            if (self.bnBeforeConv):
                y = self.convBnList1[i](y)
            if (self.bottleDrop):
                y = self.drop(y)
                
            y = self.convList[i](y)
            if (self.bn):
                y = self.bnList[i](y)
            y = self.activation(y)
            if (self.bnBeforeConv):
                y = self.convBnList2[i](y)
            if (self.convDrop):
                y = self.drop(y)
             
            x = torch.cat((x, y), dim = 1)
        return x
    
    
class DenseBlockChanged(nn.Module):
    def __init__(self, initSize, numBlocks, blockSize, bottleneckSize, 
                 dropRegime = "None", pDrop = 0, bottleDrop = True, convDrop = False, 
                 bn = True, nonlin = 'Relu', bottleBn = True, bottleNonlin = True, bnRegime = "bn", 
                 bnBeforeConv = False, bias = False):
        super().__init__()
        self.initSize = initSize
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.bottleDrop = bottleDrop
        self.convDrop = convDrop
        self.bn = bn
        self.bias = bias
        self.nonlin = nonlin
        self.pDrop = pDrop
        self.bottleBn = bottleBn
        self.bottleNonlin = bottleNonlin
        
        assert dropRegime in ["None", "Drop", "ColumnDrop"]
        assert nonlin in ["Identity", "Relu", "MaxMin", "MaxMinPairwise"]
        assert bnRegime in ["bn", "ln"]
        
        if (bnRegime == "bn"):
            bnModule = nn.BatchNorm2d
        else:
            bnModule = LinearTransform
        
        self.vertConvList = nn.ModuleList()
        self.convList = nn.ModuleList()
        if (self.bn):
            self.bnList = nn.ModuleList()
        
        if (self.bottleBn):
            self.bottleBnList = nn.ModuleList()
        
        self.drop = Identity()
        if (dropRegime == "Drop"):
            self.drop = nn.Dropout(self.pDrop)
        if (dropRegime == "ColumnDrop"):
            self.drop = ColumnDrop(self.pDrop, 0)
        
        if (self.nonlin == "Identity"):
            self.activation = Identity()
        if (self.nonlin == "Relu"):
            self.activation = F.relu
        if (self.nonlin == "MaxMin"):
            self.activation = maxmin_fc
        if (self.nonlin == "MaxMinPairwise"):
            self.activation = MaxMinPairwise
        
        if (nonlin == "MaxMin"):
            mainSizeInc = 2 * blockSize
        else:
            mainSizeInc = blockSize
            
        convSizeInp = bottleneckSize
        if (bottleNonlin and nonlin == "MaxMin"):
            convSizeInp = 2 * bottleneckSize
        
        size = self.initSize
        for i in range(numBlocks):
            if (self.bottleBn):
                self.bottleBnList.append(bnModule(num_features = size))
            self.vertConvList.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            
            if (self.bn):
                self.bnList.append(bnModule(num_features = convSizeInp))
            self.convList.append(nn.Conv2d(convSizeInp, blockSize, 3, padding = 1, bias = self.bias))
            size += blockSize
        
        self.finalSize = size
            
    def forward(self, x):
        for i in range(self.numBlocks):
            if (self.bottleBn):
                y = self.bottleBnList[i](x)
            if (self.bottleNonlin):
                y = self.activation(y)
            y = self.vertConvList[i](y)
            
            if (self.bottleDrop):
                y = self.drop(y)
            
            if (self.bn):
                y = self.bnList[i](y)
            y = self.activation(y)
                            
            y = self.convList[i](y)
           
            if (self.convDrop):
                y = self.drop(y)
             
            x = torch.cat((x, y), dim = 1)
        return x
            
        
        
class InputDependentTransition(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, padding = 1)
    
    def forward(self, x):
        mask = self.conv(x)
        m_sh = mask.shape
        mask = F.softmax(mask.view(m_sh[0], m_sh[1], -1), dim = 2).view(m_sh)
        return x.mul(mask).view(x.shape[0], x.shape[1], -1).mean(dim = 2)
   
    
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, nonlin = "Relu", 
                 dropRegime = "None", pDrop = 0, finalPool = False, bnRegime = "bn", bias = False):
        super().__init__()
        self.bn = bn
        self.nonlin = nonlin
        self.bias = bias
        self.pDrop = pDrop
        self.finalPool = finalPool
        if (bnRegime == "bn"):
            bnM = nn.BatchNorm2d
        else:
            bnM = LinearTransform
        
        assert nonlin in ["Relu", "Identity", "MaxMinPairwise"]
        
        if (self.nonlin == "Identity"):
            self.activation = Identity()
        if (self.nonlin == "Relu"):
            self.activation = F.relu
        if (self.nonlin == "MaxMinPairwise"):
            self.activation = MaxMinPairwise
        
        self.drop = Identity()
        if (dropRegime == "Drop"):
            self.drop = nn.Dropout(self.pDrop)
        if (dropRegime == "ColumnDrop"):
            self.drop = ColumnDrop(self.pDrop, 0)
        
        if (self.finalPool):
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        else:
            if (self.bn):
                self.bnModule = bnM(num_features = out_channels)
            self.vertConv = nn.Conv2d(in_channels, out_channels, 1, padding = 0, bias = self.bias)
            self.pool = nn.AvgPool2d(2, 2)
        
        
    def forward(self, x):
        if (not self.finalPool):
            x = self.vertConv(x)
            if (self.bn):
                x = self.bnModule(x)
            x = self.activation(x)
        x = self.drop(x)
        return self.pool(x)
    
    
    
class TransitionChanged(nn.Module):
    def __init__(self, in_channels, out_channels, bn = True, nonlin = "Relu", 
                 dropRegime = "None", pDrop = 0, finalPool = False, bnRegime = "bn", bias = False):
        super().__init__()
        self.bn = bn
        self.nonlin = nonlin
        self.bias = bias
        self.pDrop = pDrop
        self.finalPool = finalPool
        if (bnRegime == "bn"):
            bnM = nn.BatchNorm2d
        else:
            bnM = LinearTransform
            
        assert nonlin in ["Relu", "Identity", "MaxMinPairwise"]
        
        if (self.nonlin == "Identity"):
            self.activation = Identity()
        if (self.nonlin == "Relu"):
            self.activation = F.relu
        if (self.nonlin == "MaxMinPairwise"):
            self.activation = MaxMinPairwise
        
        self.drop = Identity()
        if (dropRegime == "Drop"):
            self.drop = nn.Dropout(self.pDrop)
        if (dropRegime == "ColumnDrop"):
            self.drop = ColumnDrop(self.pDrop, 0)
            
        if (self.bn):
            self.bnModule = bnM(num_features = in_channels)
        
        if (self.finalPool):
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.vertConv = nn.Conv2d(in_channels, out_channels, 1, padding = 0, bias = self.bias)
            self.pool = nn.AvgPool2d(2, 2)
        
        
    def forward(self, x):
        if (self.bn):
            x = self.bnModule(x)
        x = self.activation(x)
        if (not self.finalPool):
            x = self.vertConv(x)
        x = self.drop(x)
        return self.pool(x)
    
            
            
            
            
            
class GeneralConvNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, poolFraction = 0.5,
                 convDropRegime = "None", bottleDrop = False, convDrop = False, pConvDrop = 0, convPoolDrop = False, 
                 pPoolDrop = 0, poolDropRegime = "None", finalDrop = False, pFinalDrop = 0, nonlin = "Relu", kwta = False, bn = True, 
                 bottleBn = True, bottleNonlin = True, transitionBn = True, transitionNonlin = True,
                 bnBeforeConv = False, bnRegime = "bn", reversedBnOrder = True, bias = True, inputDependentPool = False):
        super().__init__()
        self.netName = "GeneralConvNet"
        self.bias = bias
        self.numDenseLayers = len(numBlocks)
        self.finalDrop = finalDrop
        
        assert convDropRegime in ["None", "Drop", "ColumnDrop"]
        assert poolDropRegime in ["None", "Drop", "ColumnDrop"]
        assert nonlin in ["Identity", "Relu", "MaxMin", "MaxMinPairwise"]
            
        self.DenseLayers = nn.ModuleList()
        self.TransitionLayers = nn.ModuleList()
        
        if (reversedBnOrder):
            transitionModule = TransitionChanged
            denseModule = DenseBlockChanged
        else:
            transitionModule = Transition
            denseModule = DenseBlock
        
        if (transitionNonlin):
            trNonLin = nonlin
        else:
            trNonLin = Identity()
        
        self.finalDropFunc = Identity()
        if (finalDrop):
            self.finalDropFunc = nn.Dropout(pFinalDrop)
        
        out_size = 2 * blockSize
        self.init_conv = nn.Conv2d(3, out_size, 3, padding = 1, bias = self.bias)            
        
        for layer in range(self.numDenseLayers):
            self.DenseLayers.append(denseModule(out_size, numBlocks[layer], blockSize, bottleneckSize, dropRegime = convDropRegime,
                                      pDrop = pConvDrop, bottleDrop = bottleDrop, convDrop = convDrop, 
                                      bn = bn, nonlin = nonlin, bottleBn = bottleBn, bottleNonlin = bottleNonlin, 
                                      bnBeforeConv = bnBeforeConv, bnRegime = bnRegime, bias = bias))
            if (layer != self.numDenseLayers - 1):
                in_size = self.DenseLayers[layer].finalSize 
                out_size = int(in_size * poolFraction)
                poolDropCoef = 0
                if (poolDropRegime != "None"):
                    poolDropCoef = pPoolDrop[layer]
                self.TransitionLayers.append(transitionModule(in_size, out_size, bn = transitionBn, nonlin = trNonLin,
                                        dropRegime = poolDropRegime, pDrop = poolDropCoef, 
                                        finalPool = False, bnRegime = bnRegime, bias = bias))
                                             
        out_size = self.DenseLayers[self.numDenseLayers-1].finalSize
        poolDropCoef = 0
        if (poolDropRegime != "None"):
            poolDropCoef = pPoolDrop[self.numDenseLayers - 1]
            
        if (inputDependentPool):
            self.TransitionLayers.append(InputDependentTransition(out_size))
        else:
            self.TransitionLayers.append(transitionModule(out_size, out_size, nonlin = trNonLin, 
                                                dropRegime = poolDropRegime, pDrop = poolDropCoef, 
                                                finalPool = True, bnRegime = bnRegime, bias = bias))
        self.fc = nn.Linear(out_size, 10, bias = self.bias)
        
    def forward(self, x):
        x = self.init_conv(x)
        for i in range(self.numDenseLayers):
            x = self.TransitionLayers[i](self.DenseLayers[i](x))
        x = x.squeeze()   
        x = self.finalDropFunc(x)
        x = self.fc(x)
        return x
        
        
        
        
        
    
    
    
#DenseBottleneckMaxMinNet 
class DenseBottleneckMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, innerNonLinearity = False, concatinateVert = False, bias = True, 
                 partialPoolings = True, vertConvPool = True, poolFraction = 0.5,
                 columnCropPool = False, dropRegime = "None", pDrop = 0, pNotDrop = 0,
                 inputDrop = False, bottleDrop = True, convDrop = True, convPoolDrop = True, innerPoolDrop = False, kwta = False,
                 bn = True):
        super().__init__()
        self.netName = "Dense Bottleneck Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.innerNonLinearity = innerNonLinearity
        self.concatinateVert = concatinateVert
        self.bias = bias        
        
        self.partialPoolings = partialPoolings
        self.vertConvPool = vertConvPool
        self.poolFraction = poolFraction
        
        self.columnCropPool = columnCropPool
        
        self.dropRegime = dropRegime
        self.pDrop = pDrop
        self.pNotDrop = pNotDrop
        
        self.inputDrop = inputDrop
        self.bottleDrop = bottleDrop
        self.convDrop = convDrop
        self.convPoolDrop = convPoolDrop
        self.innerPoolDrop = innerPoolDrop
        
        self.kwta = kwta
        
        self.bn = bn
        
        if (not (self.dropRegime in ["None", "Drop", "ColumnDrop", "ColumnCrop"])):
            print("Error: drop regime is not allowed")
        
        if (self.dropRegime != "None" and pDrop == 0):
            print("Error: drop with zero probability")
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        if (self.bn):
            self.bnList1 = nn.ModuleList()
            self.bnList2 = nn.ModuleList()
            self.bnList3 = nn.ModuleList()
        
        
        self.drop = Identity()
        if (dropRegime == "Drop"):
            self.drop = nn.Dropout(self.pDrop)
        if (dropRegime == "ColumnDrop"):
            self.drop = ColumnDrop(self.pDrop, self.pNotDrop)
        if (dropRegime == "ColumnCrop"):
            self.drop = ColumnCrop(1.0 - self.pDrop)
        
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            if (self.innerNonLinearity):
                self.convList1.append(nn.Conv2d(2 * bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            else:
                self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            if (self.concatinateVert):
                size += 2 * blockSize + 2 * bottleneckSize
            else:
                size += 2 * blockSize
            if (self.bn):
                self.bnList1.append(nn.BatchNorm2d(num_features = blockSize))
        
        fullSize = size
        
        if (self.partialPoolings):
            size = 2 * blockSize
            
            
        if (self.vertConvPool):
            size = int(fullSize * self.poolFraction)
            self.vertPool1 = nn.Conv2d(fullSize, size, 1, padding = 0, bias = self.bias)
            
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            if (self.innerNonLinearity):
                self.convList2.append(nn.Conv2d(2 * bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            else:
                self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            if (self.concatinateVert):
                size += 2 * blockSize + 2 * bottleneckSize
            else:
                size += 2 * blockSize
            if (self.bn):
                self.bnList2.append(nn.BatchNorm2d(num_features = blockSize))
            
        fullSize = size
            
        if (self.partialPoolings):
            size = 2 * blockSize
            
        if (self.vertConvPool):
            size = int(fullSize * self.poolFraction)
            self.vertPool2 = nn.Conv2d(fullSize, size, 1, padding = 0, bias = self.bias)
            
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            if (self.innerNonLinearity):
                self.convList3.append(nn.Conv2d(2 * bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            else:
                self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            if (self.concatinateVert):
                size += 2 * blockSize + 2 * bottleneckSize
            else:
                size += 2 * blockSize
            if (self.bn):
                self.bnList3.append(nn.BatchNorm2d(num_features = blockSize))
            
        fullSize = size
        if (self.partialPoolings):
            size = 2 * blockSize
        
        #if (self.vertConvPool):
        #    size = int(fullSize * self.poolFraction)
        #    self.vertPool3 = nn.Conv2d(fullSize, size, 1, padding = 0, bias = self.bias)
        
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        if (self.columnCropPool):
            self.pool3 = FullColumnCropPool(1.0-self.pDrop)
            
        if (self.kwta):
            self.kwta_soft = KWinnersTakeAllSoft()
        
        self.fc = nn.Linear(fullSize, 10, bias = self.bias)
        
        
    
    def forward(self, x):
        if (self.inputDrop):
            x = self.drop(x)
            
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            if (self.bottleDrop):
                y = self.drop(y)
            if (self.innerNonLinearity):
                y = maxmin_fc(y)
            if (self.concatinateVert):
                x = torch.cat((x, y), dim = 1)
            y = self.convList1[i](y)
            if (self.convDrop):
                y = self.drop(y)
            if (self.bn):
                y = self.bnList1[i](y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        if (self.vertConvPool):
            x = self.vertPool1(x)
            if (self.convPoolDrop):
                x = self.drop(x)
            x = self.pool1(x)
            
        else:
            if (self.partialPoolings):                
                x = self.pool1(y)
            else:
                x = self.pool1(x)           
        
        
        if (self.innerPoolDrop):
            x = self.drop(x)
            
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            if (self.bottleDrop):
                y = self.drop(y)
            if (self.innerNonLinearity):
                y = maxmin_fc(y)
            if (self.concatinateVert):
                x = torch.cat((x, y), dim = 1)
            y = self.convList2[i](y)
            if (self.convDrop):
                y = self.drop(y)
            if (self.bn):
                y = self.bnList2[i](y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
            
            
        if (self.vertConvPool):
            x = self.vertPool2(x)
            if (self.convPoolDrop):
                x = self.drop(x)
            x = self.pool2(x)
            
        else:
            if (self.partialPoolings):                
                x = self.pool2(y)
            else:
                x = self.pool2(x) 
                
                
        
        if (self.innerPoolDrop):
            x = self.drop(x)
            
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            if (self.bottleDrop):
                y = self.drop(y)
            if (self.innerNonLinearity):
                y = maxmin_fc(y)
            if (self.concatinateVert):
                x = torch.cat((x, y), dim = 1)
            y = self.convList3[i](y)
            if (self.convDrop):
                y = self.drop(y)
            if (self.bn):
                y = self.bnList3[i](y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
        
        if (self.vertConvPool):
            #x = self.vertPool3(x)
            if (self.convPoolDrop):
                x = self.drop(x)
            x = self.pool3(x)
            
        else:
            if (self.partialPoolings):                
                x = self.pool3(y)
            else:
                x = self.pool3(x)
        
        if (self.dropRegime == "Drop"):
            x = self.drop(x)
        
        x = x.squeeze()
        
        if (self.kwta):
            x = self.kwta_soft(x)
        
        x = self.fc(x)
        return x
        
        
        
        

def getMultipliers(y, alpha, pDrop):
    passive = y.new().resize_as_(y)
    updown = y.new().resize_as_(y)
    passive.bernoulli_(pDrop)
    updown.bernoulli_(0.5)
    result = 1.0 + alpha * passive * (2.0 * updown - 1.0)
    return result

def setColumnDropMask(y, pDrop):
    active = y.new(y.shape[0], 1, y.shape[2], y.shape[3])
    active.bernoulli_(1.0 - pDrop).mul_(1.0/(1.0 - pDrop))
    return active

def columnDrop(y, pDrop):
    active = y.new(y.shape[0], 1, y.shape[2], y.shape[3])
    active.bernoulli_(1.0 - pDrop).mul_(1.0/(1.0 - pDrop))
    y.mul_(active) 
        
#DenseBottleneckAlphaMaxMinNet 
class DenseBottleneckAlphaMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, alpha, pDrop):
        super().__init__()
        self.netName = "Dense Bottleneck Alpha Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.alpha = alpha
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            y = self.convList1[i](y)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            x = torch.cat((x, maxmin_fc(y)), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            y = self.convList2[i](y)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            x = torch.cat((x, maxmin_fc(y)), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            y = self.convList3[i](y)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            x = torch.cat((x, maxmin_fc(y)), dim = 1)
        x = self.pool3(x)
        x = x.view(-1, 3 + 6 * self.numBlocks * self.blockSize)
        x.mul_(getMultipliers(x, self.alpha, self.pDrop))
        x = self.fc(x)
        return x
    
    
    
class DenseBottleneckSynchronizedColumnDropMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, pDrop):
        super().__init__()
        self.netName = "Dense Bottleneck Synchronized Column Drop Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
        
        size = 2 * blockSize
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
        
        size = 2 * blockSize
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(2 * self.blockSize, 10)
        
    def forward(self, x):
        if (self.training):
            mask = setColumnDropMask(x, self.pDrop)
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            y = self.convList1[i](y)
            if (self.training):
                y.mul_(mask)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool1(y)
        if (self.training):
            mask = setColumnDropMask(x, self.pDrop)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            y = self.convList2[i](y)
            if (self.training):
                y.mul_(mask)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool2(y)
        if (self.training):
            mask = setColumnDropMask(x, self.pDrop)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            y = self.convList3[i](y)
            if (self.training):
                y.mul_(mask)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool3(y)
        x = x.view(-1, 2 * self.blockSize)
        x = self.fc(x)
        return x
    
    

class DenseBottleneckDropMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, pDrop):
        super().__init__()
        self.netName = "Dense Bottleneck Drop Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        self.drop = nn.Dropout(self.pDrop)
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
        
        size = 2 * blockSize
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
        
        size = 2 * blockSize
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(2 * self.blockSize, 10)
        
    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            y = self.convList1[i](y)
            y = drop(y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool1(y)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            y = self.convList2[i](y)
            y = drop(y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool2(y)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            y = self.convList3[i](y)
            y = drop(y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool3(y)
        x = x.view(-1, 2 * self.blockSize)
        x = drop(x)
        x = self.fc(x)
        return x
    
    
class DenseBottleneckColumnDropMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, pDrop):
        super().__init__()
        self.netName = "Dense Bottleneck Column Drop Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
        
        size = 2 * blockSize
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
        
        size = 2 * blockSize
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size += 2 * blockSize
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(2 * self.blockSize, 10)
        
    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            y = self.convList1[i](y)
            if (self.training):
                columnDrop(y, self.pDrop)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool1(y)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            y = self.convList2[i](y)
            if (self.training):
                columnDrop(y, self.pDrop)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool2(y)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            y = self.convList3[i](y)
            if (self.training):
                columnDrop(y, self.pDrop)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
        x = self.pool3(y)
        x = x.view(-1, 2 * self.blockSize)
        x = self.fc(x)
        return x
    
    
    
    
class SequentialBottleneckMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize):
        super().__init__()
        self.netName = "Sequential Bottleneck Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size = 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            y = self.convList1[i](y)
            x = maxmin_fc(y)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            y = self.convList2[i](y)
            x = maxmin_fc(y)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            y = self.convList3[i](y)
            x = maxmin_fc(y)
        x = self.pool3(x)
        x = x.view(-1, 2 * self.blockSize)
        x = self.fc(x)
        return x
    
    
    
class SequentialPyramidalBottleneckMaxMinNet(BaseNet):
    def __init__(self, blockSize, bottleneckSize):
        super().__init__()
        self.netName = "Sequential Pyramidal Bottleneck Max Min Net"
        self.numBlocks = 2
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(self.numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 0))
            size = 2 * blockSize
            
        for i in range(self.numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 0))
            
        for i in range(self.numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 0))
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            y = self.convList1[i](y)
            x = maxmin_fc(y)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            y = self.convList2[i](y)
            x = maxmin_fc(y)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            y = self.convList3[i](y)
            x = maxmin_fc(y)
        x = x.view(x.shape[0], 2 * self.blockSize)
        x = self.fc(x)
        return x
    
    
    
    
    
#DenseBottleneckMaxMinNet 
class SequentialBottleneckAlphaMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, alpha, pDrop):
        super().__init__()
        self.netName = "Sequential Bottleneck Alpha Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.alpha = alpha
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size = 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            y = self.convList1[i](y)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            x = maxmin_fc(y)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            y = self.convList2[i](y)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            x = maxmin_fc(y)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            y = self.convList3[i](y)
            if (self.training):
                y.mul_(getMultipliers(y, self.alpha, self.pDrop))
            x = maxmin_fc(y)
        x = self.pool3(x)
        x = x.view(-1, 2 * self.blockSize)
        x = self.fc(x)
        return x
    
    
    
#DenseBottleneckMaxMinNet 
class SequentialBottleneckColumnDropMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, pDrop):
        super().__init__()
        self.netName = "Sequential Bottleneck Column Drop Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.vertConvList1 = nn.ModuleList()
        self.vertConvList2 = nn.ModuleList()
        self.vertConvList3 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.vertConvList1.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            size = 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1))
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            y = self.convList1[i](y)
            if (self.training):
                columnDrop(y, self.pDrop)
            x = maxmin_fc(y)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            y = self.vertConvList2[i](x)
            y = self.convList2[i](y)
            if (self.training):
                columnDrop(y, self.pDrop)
            x = maxmin_fc(y)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            y = self.vertConvList3[i](x)
            y = self.convList3[i](y)
            if (self.training):
                columnDrop(y, self.pDrop)
            x = maxmin_fc(y)
        x = self.pool3(x)
        x = x.view(-1, 2 * self.blockSize)
        x = self.fc(x)
        return x
    
    

#DenseBottleneckAlphaMaxMinNet 
class SingleDenseColumnDropMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, pDrop):
        super().__init__()
        self.netName = "Single Dense Bottleneck Column Drop Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.pDrop = pDrop
        
        self.convList1 = nn.ModuleList()
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1))
            size += 2 * self.blockSize
            
        self.pool1 = nn.AvgPool2d(32, 32)
        self.fc = nn.Linear(2 * self.blockSize, 10)
        
    def forward(self, x):
        for i in range(self.numBlocks):
            y = maxmin_fc(self.convList1[i](x))
            x = torch.cat((x, y), dim = 1)
        if (self.training):
                columnDrop(y, self.pDrop)
        x = self.pool1(y)
        x = x.view(-1, 2 * self.blockSize)
        x = self.fc(x)
        return x
    