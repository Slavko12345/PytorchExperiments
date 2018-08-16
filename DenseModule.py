#DenseReluNet 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.autograd import Variable
from torch.nn.parameter import Parameter

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
            self.Basis = Parameter(torch.Tensor([[[1, -1, 0], [-1, 1, 0], [0, 0, 0]], 
                                                 [[0, -1, 1], [0, 1, -1], [0, 0, 0]], 
                                                 [[0, 0, 0], [-1, 1, 0], [1, -1, 0]],
                                                 [[0, 0, 0], [0, 1, -1], [0, -1, 1]]]), requires_grad = False)
       
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
            kernel = torch.einsum("ijk,kmn->ijmn", (self.convParams, self.Basis))
        
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
        if (torch.bernoulli(torch.tensor([self.pNotDrop])).item()):
            return x
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
    
    
    
#DenseBottleneckMaxMinNet 
class DenseBottleneckMaxMinNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckSize, bias = True, 
                 partialPoolings = True, columnCropPool = False, dropRegime = "None", pDrop = 0, pNotDrop = 0,
                 inputDrop = False, bottleDrop = True, convDrop = True, innerPoolDrop = True):
        super().__init__()
        self.netName = "Dense Bottleneck Max Min Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.bottleneckSize = bottleneckSize
        self.bias = bias
        self.partialPoolings = partialPoolings
        self.columnCropPool = columnCropPool
        
        self.dropRegime = dropRegime
        self.pDrop = pDrop
        self.pNotDrop = pNotDrop
        
        self.inputDrop = inputDrop
        self.bottleDrop = bottleDrop
        self.convDrop = convDrop
        self.innerPoolDrop = innerPoolDrop
        
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
            self.convList1.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            size += 2 * blockSize
        
        if (self.partialPoolings):
            size = 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList2.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            self.convList2.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            size += 2 * blockSize
            
        if (self.partialPoolings):
            size = 2 * blockSize
            
        for i in range(numBlocks):
            self.vertConvList3.append(nn.Conv2d(size, bottleneckSize, 1, padding = 0, bias = self.bias))
            self.convList3.append(nn.Conv2d(bottleneckSize, blockSize, 3, padding = 1, bias = self.bias))
            size += 2 * blockSize
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        if (self.columnCropPool):
            self.pool3 = FullColumnCropPool(1.0-self.pDrop)
        
        if (self.partialPoolings):
            size = 2 * blockSize
        self.fc = nn.Linear(size, 10, bias = self.bias)

    def forward(self, x):
        if (self.inputDrop):
            x = self.drop(x)
            
        for i in range(self.numBlocks):
            y = self.vertConvList1[i](x)
            if (self.bottleDrop):
                y = self.drop(y)
            y = self.convList1[i](y)
            if (self.convDrop):
                y = self.drop(y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
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
            y = self.convList2[i](y)
            if (self.convDrop):
                y = self.drop(y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
            
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
            y = self.convList3[i](y)
            if (self.convDrop):
                y = self.drop(y)
            y = maxmin_fc(y)
            x = torch.cat((x, y), dim = 1)
        
        if (self.partialPoolings):
            x = self.pool3(y)
        else:
            x = self.pool3(x)
        
        if (self.dropRegime == "Drop"):
            x = self.drop(x)
        
        if (self.partialPoolings):
            x = x.view(-1, 2 * self.blockSize)
        else:
            x = x.view(-1, 3 + 6 * self.numBlocks * self.blockSize)
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
    