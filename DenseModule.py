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
    
    def numTrainableParams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def printNet(self):
        print(self.netName + " will be trained!")
        print("Number of parameters: ", self.numParams())
        print("Number of trainable parameters: ", self.numTrainableParams())
    
    def getLoss(self, reduction = "mean"):
        return nn.CrossEntropyLoss(reduction = reduction)
    
    def convertLabels(self, labels):
        return labels
    

class OctaveNetImproved(BaseNet):
    def __init__(self, numBlocks, blockSize, pPoolDrop = [0,0,0], pOctaveDrop = 0, 
                 reduction = "average", numPoolings = 3, initDepth = 3, numClasses = 10):
        super().__init__()
        assert len(pPoolDrop) == numPoolings
        self.netName = "Octave Net Improved"
        self.numBlocks, self.blockSize, self.numPoolings = numBlocks, blockSize, numPoolings
        
        self.orientationPool = {"average": averageMultiple,
                                "max": maxMultiple,
                                "square": sumOfSquaresMultiple}
        
        self.convs3_3, self.convs1d, self.poolDrop, self.pools = \
        nn.ModuleList(), nn.ModuleList(), nn.ModuleList()  , nn.ModuleList() 
        
        for _ in range(numPoolings - 1):
            self.pools.append(nn.AvgPool2d(2))
        self.pools.append(nn.AdaptiveAvgPool2d(1))
        
        for layer in range(numPoolings):
            self.convs3_3.append(nn.ModuleList())
            self.convs1d.append(nn.ModuleList())
            self.poolDrop.append(OctaveColumnDrop(pPoolDrop[layer]))
        
        size = initDepth
        self.initConv = ImageToOctaveEfficientConv(size, blockSize)
        size += blockSize
        
        for layer in range(numPoolings):
            for block in range(numBlocks):
                self.convs1d[layer].append(OctaveToOctaveEfficientConv1d(size, blockSize))
                self.convs3_3[layer].append(OctaveToOctaveEfficientConv(blockSize, blockSize))                
                size += blockSize
        
        self.octDrop = OctaveDrop(pOctaveDrop)
        self.octavePool = self.orientationPool[reduction]
        
        self.fc = nn.Linear(size, numClasses)
        
    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y)
        
        for layer in range(self.numPoolings):
            for block in range(self.numBlocks):
                y = F.relu(self.convs1d[layer][block](x))
                y = self.convs3_3[layer][block](y)
                x = catReluMultipleMultiple(x, y)  
            x = self.poolDrop[layer](x)
            x = self.pools[layer](x)
            
        x = x.view(*x.shape[:2])
        x = self.octDrop(x)
        x = averageMultiple(x)
        x = self.fc(x)
        return x
    
class QuadNetImproved(BaseNet):
    def __init__(self, numBlocks, blockSize, pPoolDrop = [0,0,0], pOctaveDrop = 0, 
                 reduction = "average", numPoolings = 3, initDepth = 3, numClasses = 10):
        super().__init__()
        assert len(pPoolDrop) == numPoolings
        self.netName = "Quad Net Improved"
        self.numBlocks, self.blockSize, self.numPoolings = numBlocks, blockSize, numPoolings
        
        self.orientationPool = {"average": averageMultiple,
                                "max": maxMultiple,
                                "square": sumOfSquaresMultiple}
        
        self.convs3_3, self.convs1d, self.poolDrop, self.pools = \
        nn.ModuleList(), nn.ModuleList(), nn.ModuleList()  , nn.ModuleList() 
        
        for _ in range(numPoolings - 1):
            self.pools.append(nn.AvgPool2d(2))
        self.pools.append(nn.AdaptiveAvgPool2d(1))
        
        for layer in range(numPoolings):
            self.convs3_3.append(nn.ModuleList())
            self.convs1d.append(nn.ModuleList())
            self.poolDrop.append(OctaveColumnDrop(pPoolDrop[layer], num = 4))
        
        size = initDepth
        self.initConv = ImageToQuadEfficientConv(size, blockSize)
        size += blockSize
        
        for layer in range(numPoolings):
            for block in range(numBlocks):
                self.convs1d [layer].append(QuadToQuadEfficientConv1d(size, blockSize))
                self.convs3_3[layer].append(QuadToQuadEfficientConv(blockSize, blockSize))                
                size += blockSize
        
        self.octDrop = OctaveDrop(pOctaveDrop, num = 4)
        self.octavePool = self.orientationPool[reduction]
        
        self.fc = nn.Linear(size, numClasses)
        
    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y, num = 4)
        
        for layer in range(self.numPoolings):
            for block in range(self.numBlocks):
                y = F.relu(self.convs1d[layer][block](x))
                y = self.convs3_3[layer][block](y)
                x = catReluMultipleMultiple(x, y, num = 4)  
            x = self.poolDrop[layer](x)
            x = self.pools[layer](x)
            
        x = x.view(*x.shape[:2])
        x = self.octDrop(x)
        x = self.octavePool(x, num = 4)
        x = self.fc(x)
        return x
    
    

    
    
    
class OrientationExtractor(nn.Module):
    def __init__(self, numBlocks, symmDepth, deltaDepth, kxyDepth, kdDepth, numPools = 3):
        super().__init__()
        self.numBlocks = numBlocks
        blockSize = symmDepth + 2 * deltaDepth + kxyDepth + kdDepth
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
        
        self.pools = nn.ModuleList()
        for i in range(numPools):
            self.pools.append(nn.AvgPool2d(2))
        
        self.finalPool = nn.AdaptiveAvgPool2d(1)
        
        size = 3
        self.initConv = SteerableImageToOctave(size, symmDepth, deltaDepth, kxyDepth, kdDepth)
        size += blockSize
        
        for layer in range(numPools):
            for i in range(numBlocks):
                self.convs[layer].append(SteerableOctaveToOctave\
                                         (size, symmDepth, deltaDepth, kxyDepth, kdDepth))
                size += blockSize
        
        self.finalFilter = SteerableKernel(1, size, 0, 1, 0, 0)
        self.finalDepth = size
        
    def getFinalDepth(self):
        return self.finalDepth
        
    def forward(self, x):
        #xInitial = x
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y, num = 8)
        for layer in range(self.numPools):
            for i in range(self.numBlocks):
                y = self.convs[layer][i](x)
                x = catReluMultipleMultiple(x, y, num = 8)
            if (layer != self.numPools - 1):
                x = self.pools[layer](x)
        x = averageMultiple(x, num = 8)
        xRepresentation = x
        
        k = self.finalFilter()
        k = k.view(k.shape[1:])
        x = F.conv2d(x, k, padding = 1)
        
        x = self.finalPool(x)
        x = x.view(x.shape[:2])
        
        vx = x[:, 0]
        vy = x[:, 1]
        
        length = torch.sqrt(vx * vx + vy * vy + 0.00000001)
        vx = vx / length
        vy = vy / length
        
        theta1 = torch.stack(( vy, -vx, torch.zeros_like(vx)), dim=-1)
        theta2 = torch.stack(( vx, vy, torch.zeros_like(vx)), dim=-1)
        theta = torch.stack((theta1, theta2), dim=-2).view(-1, 2, 3)
        
        xReprAugmented = F.pad(xRepresentation, (2, 2, 2, 2))
        #grid = F.affine_grid(theta, xInitial.shape)
        #result = F.grid_sample(F.pad(xInitial, (8, 8, 8, 8)), grid)
        
        grid = F.affine_grid(theta, xReprAugmented.shape)
        result = F.grid_sample(xReprAugmented, grid)
        return result
        
class RotationSpatialTransformerNet(BaseNet):
    def __init__(self, numBlocksO, octaveConfO, numBlocksC, pairConfigC, numPoolsO = 3):
        super().__init__()
        self.transformer = OrientationExtractor(numBlocksO, *octaveConfO, numPoolsO)
        finalDepth = self.transformer.getFinalDepth()
        self.classifier = SteerablePairNet(numBlocksC, *pairConfigC, numPools = 1, initDepth = finalDepth)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x
    

    
    
    
class QuadOrientationExtractor(BaseNet):
    def __init__(self, numBlocks, blockSize, reduction = "average", numPoolings = 3, initDepth = 3):
        super().__init__()
        self.netName = "Quad Orientation Extractor Improved"
        self.numBlocks, self.blockSize, self.numPoolings = numBlocks, blockSize, numPoolings
        
        self.orientationPool = {"average": averageMultiple,
                                "max": maxMultiple,
                                "square": sumOfSquaresMultiple}
        
        self.convs3_3, self.convs1d, self.pools = \
        nn.ModuleList(), nn.ModuleList(), nn.ModuleList() 
        
        for _ in range(numPoolings - 1):
            self.pools.append(nn.AvgPool2d(2))
        #self.pools.append(nn.AdaptiveAvgPool2d(1))
        
        for layer in range(numPoolings):
            self.convs3_3.append(nn.ModuleList())
            self.convs1d.append(nn.ModuleList())
        
        size = initDepth
        self.initConv = ImageToQuadEfficientConv(size, blockSize)
        size += blockSize
        
        for layer in range(numPoolings):
            for block in range(numBlocks):
                self.convs1d [layer].append(QuadToQuadEfficientConv1d(size, blockSize))
                self.convs3_3[layer].append(QuadToQuadEfficientConv(blockSize, blockSize))                
                size += blockSize
                
        self.finalDepth = size
        self.finalFilter = SteerableKernel(1, size, 0, 1, 0, 0)
        self.finalPool = nn.AdaptiveAvgPool2d(1)
                
    def getFinalDepth(self):
        return self.finalDepth
        
    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y, num = 4)
        
        for layer in range(self.numPoolings):
            for block in range(self.numBlocks):
                y = F.relu(self.convs1d[layer][block](x))
                y = self.convs3_3[layer][block](y)
                x = catReluMultipleMultiple(x, y, num = 4) 
            if (layer != self.numPoolings - 1):
                x = self.pools[layer](x)
        x = averageMultiple(x, num = 4)
        
        xRepresentation = x
        
        k = self.finalFilter()
        k = k.view(k.shape[1:])
        x = F.conv2d(x, k, padding = 1)
        
        x = self.finalPool(x)
        x = x.view(x.shape[:2])
        
        vx = x[:, 0]
        vy = x[:, 1]
        
        length = torch.sqrt(vx * vx + vy * vy + 0.00000001)
        vx = vx / length
        vy = vy / length
        
        theta1 = torch.stack(( vy, -vx, torch.zeros_like(vx)), dim=-1)
        theta2 = torch.stack(( vx, vy, torch.zeros_like(vx)), dim=-1)
        theta = torch.stack((theta1, theta2), dim=-2).view(-1, 2, 3)
        
        xReprAugmented = F.pad(xRepresentation, (2, 2, 2, 2))
        #grid = F.affine_grid(theta, xInitial.shape)
        #result = F.grid_sample(F.pad(xInitial, (8, 8, 8, 8)), grid)
        
        grid = F.affine_grid(theta, xReprAugmented.shape)
        result = F.grid_sample(xReprAugmented, grid)
        return result
    
    
class QuadSpatialTransformerNet(BaseNet):
    def __init__(self, numBlocksO, blockSizeO, numBlocksC, blockSizeC, numPoolsO = 3, initDepth = 3):
        super().__init__()
        self.transformer = QuadOrientationExtractor(numBlocksO, blockSizeO, numPoolsO, initDepth = initDepth)
        finalDepth = self.transformer.getFinalDepth()
        self.classifier = UsualDenseNet(numBlocksC, blockSizeC, numPools = 1, initDepth = finalDepth)
        
    def forward(self, x):
        x = self.transformer(x)
        x = self.classifier(x)
        return x
    
    
    
    
    
    
def oddRelu(x):
    return torch.max(x-1,0)+torch.min(x+1,0)


        

class SteerableImageToPairOddConv(nn.Module):
    def __init__(self, inputDepth, outDepth):
        super().__init__()
        self.symmetricKernel  = Parameter(torch.randn(6, outDepth, inputDepth)* 0.5 / inputDepth)
        self.asymmetricKernel = Parameter(torch.randn(3, outDepth, inputDepth)* 0.5 / inputDepth)
        zeros = torch.zeros(outDepth, inputDepth, 3)
        self.register_buffer("zeros", zeros)
        
    def forward(self, x):
        s = self.symmetricKernel
        a = self.asymmetricKernel
        k1 = torch.stack((s[0], s[1], s[2]), dim = 2)
        k2 = torch.stack((s[3], s[4], s[5]), dim = 2)
        ks = torch.stack((k1, k2, k1), dim = 3)
        
        k1 = torch.stack((a[0], a[1], a[2]), dim = 2)
        ka = torch.stack((k1, self.zeros, -k1), dim = 3)
        k = torch.cat((ks, ka), dim = 0)

        y = F.conv2d(x, k, padding = 1)
        return y             
        

class SteerablePairOddConv(nn.Module):
    def __init__(self, inputDepth, outputSymm, outputAsymm):
        super().__init__()
        self.sizes = [outputSymm, outputSymm, outputAsymm, outputAsymm]
        self.sizes_mixed = [outputSymm, outputAsymm, outputSymm, outputAsymm]
        self.symmetricKernel  = Parameter(torch.randn(6, 2 * outputSymm, inputDepth)* 0.5 / inputDepth)
        self.asymmetricKernel = Parameter(torch.randn(3, 2 * outputAsymm, inputDepth)* 0.5 / inputDepth)
        zeros = torch.zeros(2 * outputAsymm, inputDepth, 3)
        self.register_buffer("zeros", zeros)
        
    def forward(self, x):
        s = self.symmetricKernel
        a = self.asymmetricKernel
        k1 = torch.stack((s[0], s[1], s[2]), dim = 2)
        k2 = torch.stack((s[3], s[4], s[5]), dim = 2)
        ks = torch.stack((k1, k2, k1), dim = 3)
        
        k1 = torch.stack((a[0], a[1], a[2]), dim = 2)
        ka = torch.stack((k1, self.zeros, -k1), dim = 3)
        
        ks1, ks2 = torch.split(ks, self.sizes[:2], dim = 0)
        ka1, ka2 = torch.split(ka, self.sizes[2:], dim = 0)
        
        k = torch.cat((ks1, ka1, ks2, ka2), dim = 0)
        y = F.conv2d(x, k, padding = 1, groups = 2)
        C_, E_, D_, F_ = torch.split(y, self.sizes_mixed, dim = 1)
        return torch.cat((C_, F_, D_, E_), dim = 1)
        
        
class SteerablePairOddNet(BaseNet):
    def __init__(self, numBlocks, outputSymm, outputAsymm, initDepth, nPools = 3, activation = "tanh"):
        super().__init__()
        self.nPools = nPools
        self.numBlocks = numBlocks        
        
        self.convs = nn.ModuleList()
        for pool in range(nPools):
            self.convs.append(nn.ModuleList())
        self.pools = nn.ModuleList()
        
        self.poolingSizes = [2] * (nPools-1) + [int(32 / (2**(nPools-1)))]
        self.initConv = SteerableImageToPairOddConv(3, initDepth)
        size = initDepth
        
        for pool in range(nPools):
            for layer in range(numBlocks):
                self.convs[pool].append(SteerablePairOddConv(size, outputSymm, outputAsymm))
                size += outputSymm + outputAsymm
            self.pools.append(nn.AvgPool2d(self.poolingSizes[pool]))
        self.fc = nn.Linear(size, 10)
        
        if (activation == "tanh"):
            self.activation = F.tanh
        elif (activation == "oddRelu"):
            self.activation = oddRelu
        else:
            raise Exception("Activation function not supported!")
        
    def forward(self, x):
        x = self.initConv(x)
        for pool in range(self.nPools):
            for layer in range(self.numBlocks):
                x = catMultipleMultiple(x, torch.tanh(self.convs[pool][layer](x)), num = 2)
            x = self.pools[pool](x)
        x = x.view(x.shape[:2])
        x = x[:, :int(x.shape[1]/2)]
        x = self.fc(x)
        return x
        
        
        
        
class AdditionInvariantConv(nn.Module):
    def __init__(self, inputDepth, outputDepth, mode, var_params):
        #modes: "plain", "whole", "channel", "horVert", "diagonal"
        super().__init__()
        self.mode = mode
        #self.var_params = var_params
        self.inputDepth = inputDepth
        self.outputDepth = outputDepth
        self.k_s = 3
        self.pad = int((self.k_s - 1)/2)
        if (mode == "diagonal"):
            self.k_s = 5
        if (mode == "plain"):
            self.a = Parameter(torch.randn(outputDepth, inputDepth, 3, 3)* 0.5 / inputDepth)
        if (mode in ["horVert", "diagonal"]):
            self.a = Parameter(torch.randn(4, outputDepth, inputDepth)* 0.5 / inputDepth)
        if (mode == "channel"):
            self.a = Parameter(torch.randn(8, outputDepth, inputDepth)* 0.5 / inputDepth)
        if (mode == "whole"):
            self.a = Parameter(torch.randn(outputDepth, 27)* 0.5 / inputDepth)
                  
        zeros = torch.zeros(outputDepth, inputDepth)* 0.5 / inputDepth
        self.register_buffer("zeros", zeros)
        
    def forward(self, x):
        a = self.a
        z = self.zeros
        pad = self.pad
        
        if (self.mode == "plain"):
            return F.pad(F.conv2d(x, a), (pad, pad, pad, pad))       
            
        if (self.mode == "diagonal"):
            k1 = torch.stack((z         , -a[0]      , a[0] + a[2] , -a[2]      , z           ), dim=2)
            k2 = torch.stack((a[0]      , a[1] - a[2], -a[1] - a[3], a[3] - a[0], a[2]        ), dim=2)
            k3 = torch.stack((-a[0]-a[1], a[2] + a[3], z           , a[0] + a[1], -a[2] - a[3]), dim=2)
            k4 = torch.stack((a[1]      , a[0] - a[3], -a[0] - a[2], a[2] - a[1], a[3]        ), dim=2)
            k5 = torch.stack((z         , -a[1]      , a[1] + a[3] , -a[3]      , z           ), dim=2)
            k = torch.stack((k1,k2,k3,k4,k5), dim = 3)
            return F.pad(F.conv2d(x, k), (pad, pad, pad, pad))
        
        if (self.mode == "horVert"):
            k1 = torch.stack((a[0]        , -a[0] - a[2]  , a[2]        ), dim = 2)
            k2 = torch.stack((-a[0] - a[1], a.sum(dim = 0), -a[2] - a[3]), dim = 2)
            k3 = torch.stack((a[1]        , -a[1] - a[3]  , a[3]        ), dim = 2)
            k = torch.stack((k1,k2,k3), dim = 3)
            return F.pad(F.conv2d(x, k), (pad, pad, pad, pad))
        
        if (self.mode == "channel"):
            k1 = torch.stack((a[0], a[1]           , a[2]), dim = 2)
            k2 = torch.stack((a[3], -a.sum(dim = 0), a[4]), dim = 2)
            k3 = torch.stack((a[5], a[6]           , a[7]), dim = 2)
            k = torch.stack((k1,k2,k3), dim = 3)
            return F.pad(F.conv2d(x, k), (pad, pad, pad, pad))
        raise Exception("Requested setting of mode and var_params is not implemented!")
            
class AdditionInvariantNet(BaseNet):
    def __init__(self, initConvDepth, numBlocks, blockSize, mode = "3", var_params = "8", numPools = 3):
        super().__init__()
        self.netName = "Usual Dense Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        self.initConvDepth = initConvDepth
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
        
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        self.initConv = AdditionInvariantConv(size, initConvDepth, mode, var_params)
        size = initConvDepth
        
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
        self.fc = nn.Linear(size, 10)
        
        
    def forward(self, x):
        x = self.initConv(x)
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        x = self.fc(x.squeeze())
        return x
    
    def freezeConvLayers(self):
        for param in self.convs.parameters():
            param.requires_grad = False
    
    
    
    
    
    
def getD4Properties():
    group_table = torch.tensor([ [0,1,2,3,4,5,6,7],
                                 [1,2,3,0,7,4,5,6],
                                 [2,3,0,1,6,7,4,5],
                                 [3,0,1,2,5,6,7,4],
                                 [4,5,6,7,0,1,2,3],
                                 [5,6,7,4,3,0,1,2],
                                 [6,7,4,5,2,3,0,1],
                                 [7,4,5,6,1,2,3,0] ])
    
    inner_permutations = torch.tensor([[0, 1, 2, 3, 4],
                                       [0, 2, 1, 3, 4],
                                       [0, 1, 2, 3, 4],
                                       [0, 2, 1, 3, 4],
                                       [0, 1, 2, 3, 4],
                                       [0, 2, 1, 3, 4],
                                       [0, 1, 2, 3, 4],
                                       [0, 2, 1, 3, 4]])
    
    inner_mult = torch.tensor([[1, 1, 1, 1, 1],
                               [1, 1,-1,-1,-1],
                               [1,-1,-1, 1, 1],
                               [1,-1, 1,-1,-1],
                               [1,-1, 1, 1,-1],
                               [1,-1,-1,-1, 1],
                               [1, 1,-1, 1,-1],
                               [1, 1, 1,-1, 1]])
    
    return group_table
    
class SteerableKernel(nn.Module):
    def __init__(self, number_of_kernels, inpDepth, symmDepth, deltaDepth, kxyDepth, kdDepth):
        super().__init__()
        self.a = Parameter(torch.randn(number_of_kernels, symmDepth, inpDepth)* 0.5 / inpDepth)
        self.b = Parameter(torch.randn(number_of_kernels, symmDepth, inpDepth)* 0.5 / inpDepth)
        self.c = Parameter(torch.randn(number_of_kernels, symmDepth, inpDepth)* 0.5 / inpDepth)
        
        self.d = Parameter(torch.randn(number_of_kernels, deltaDepth, inpDepth)* 0.5 / inpDepth)
        self.e = Parameter(torch.randn(number_of_kernels, deltaDepth, inpDepth)* 0.5 / inpDepth)
        
        self.f = Parameter(torch.randn(number_of_kernels, kxyDepth, inpDepth)* 0.5 / inpDepth)
        
        self.g = Parameter(torch.randn(number_of_kernels, kdDepth, inpDepth)* 0.5 / inpDepth)
        
        self.zeros_delta = Variable(torch.zeros(number_of_kernels, deltaDepth, inpDepth, 3), requires_grad=False)        
        self.zeros_xy = Variable(torch.zeros(number_of_kernels, kxyDepth, inpDepth), requires_grad=False)
        self.zeros_d1 = Variable(torch.zeros(number_of_kernels, kdDepth, inpDepth), requires_grad=False)
        self.zeros_d3 = Variable(torch.zeros(number_of_kernels, kdDepth, inpDepth, 3), requires_grad=False)
        if torch.cuda.is_available():
            self.zeros_delta = self.zeros_delta.cuda()
            self.zeros_xy = self.zeros_xy.cuda()
            self.zeros_d1 = self.zeros_d1.cuda()
            self.zeros_d3 = self.zeros_d3.cuda()
            
    def forward(self):
        k1 = torch.stack((self.a, self.b, self.a), dim=3)
        k2 = torch.stack((self.b, self.c, self.b), dim=3)
        kernel_s = torch.stack((k1, k2, k1), dim = 4)
        
        k1 = torch.stack((self.d, self.e, self.d), dim=3)
        kernel_x = torch.stack((-k1, self.zeros_delta, k1), dim = 4)
        kernel_y = -kernel_x.transpose(-1, -2)
        
        k1 = torch.stack((self.zeros_xy, -self.f, self.zeros_xy), dim=3)
        k2 = torch.stack((self.f, self.zeros_xy, self.f), dim=3)
        kernel_xy = torch.stack((k1, k2, k1), dim = 4)
        
        k1 = torch.stack((self.g, self.zeros_d1, -self.g), dim=3)
        kernel_d = torch.stack((k1, self.zeros_d3, -k1), dim = 4)
        
        kernel = torch.cat((kernel_s, kernel_x, kernel_y, kernel_xy, kernel_d), dim = 1)  
        
        return kernel
                               
class SteerableD4Kernel(nn.Module):
    def __init__(self, symmDepth, deltaDepth, kxyDepth, kdDepth):
        super().__init__()
        self.sizes = [symmDepth, deltaDepth, deltaDepth, kxyDepth, kdDepth]
        
    def forward(self, kernel):
        ks, kx, ky, kxy, kd = torch.split(kernel, self.sizes, dim = 1)
        kernel0 = torch.cat((ks[0], kx[0], ky[0], kxy[0], kd[0]), dim = 0)
        kernel1 = torch.cat((ks[1], ky[1],-kx[1],-kxy[1],-kd[1]), dim = 0)
        kernel2 = torch.cat((ks[2],-kx[2],-ky[2], kxy[2], kd[2]), dim = 0)
        kernel3 = torch.cat((ks[3],-ky[3], kx[3],-kxy[3],-kd[3]), dim = 0)
        kernel4 = torch.cat((ks[4],-kx[4], ky[4], kxy[4],-kd[4]), dim = 0)
        kernel5 = torch.cat((ks[5],-ky[5],-kx[5],-kxy[5], kd[5]), dim = 0)
        kernel6 = torch.cat((ks[6], kx[6],-ky[6], kxy[6],-kd[6]), dim = 0)
        kernel7 = torch.cat((ks[7], ky[7], kx[7],-kxy[7], kd[7]), dim = 0)
        return torch.cat((kernel0, kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7), dim=0)
        
class SteerableD4Tensor(nn.Module):
    def __init__(self, symmDepth, deltaDepth, kxyDepth, kdDepth):
        super().__init__()
        self.sizes = [symmDepth, deltaDepth, deltaDepth, kxyDepth, kdDepth]
        #self.sizes = [size for size in self.sizes if size != 0]
        
    def forward(self, T0):
        A, B, C, D, E = torch.split(T0, self.sizes, dim = 1)
        T1 = torch.cat((A, C,-B,-D,-E), dim = 1)
        T2 = torch.cat((A,-B,-C, D, E), dim = 1)
        T3 = torch.cat((A,-C, B,-D,-E), dim = 1)
        T4 = torch.cat((A,-B, C, D,-E), dim = 1)
        T5 = torch.cat((A,-C,-B,-D, E), dim = 1)
        T6 = torch.cat((A, B,-C, D,-E), dim = 1)
        T7 = torch.cat((A, C, B,-D, E), dim = 1)
        
        return torch.cat((T0, T1, T2, T3, T4, T5, T6, T7), dim = 1)
        
        
class SteerableImageToOctave(nn.Module):
    def __init__(self, inputDepth, symmDepth, deltaDepth, kxyDepth, kdDepth):
        super().__init__()
        self.kernel = SteerableKernel(1, inputDepth, symmDepth, deltaDepth, kxyDepth, kdDepth)
        self.sizes = [symmDepth, deltaDepth, deltaDepth, kxyDepth, kdDepth]
        self.steer = SteerableD4Tensor(symmDepth, deltaDepth, kxyDepth, kdDepth)
        
    def forward(self, x):
        k = self.kernel()[0]
        T0 = F.conv2d(x, k, padding = 1)
        return self.steer(T0)
        
        
class SteerableOctaveToOctave(nn.Module):
    def __init__(self, inputDepth, symmDepth, deltaDepth, kxyDepth, kdDepth):
        super().__init__()
        self.inputDepth = inputDepth
        self.outputDepth = symmDepth + 2 * deltaDepth + kxyDepth + kdDepth
        self.kernels = SteerableKernel(8, inputDepth, symmDepth, deltaDepth, kxyDepth, kdDepth)
        self.getSteerableKernel = SteerableD4Kernel(symmDepth, deltaDepth, kxyDepth, kdDepth)
        
        self.group_table = getD4Properties()
        self.kernel_indices = torch.sort(self.group_table.t(), dim = 1)[1]
        
    def forward(self, x):
        k = self.kernels()
        k_full = k[self.kernel_indices]
        k_full = k_full.transpose(1,2)
        k_full = self.getSteerableKernel(k_full)
        k_full = k_full.contiguous().view(8 * self.outputDepth, 8 * self.inputDepth, 3, 3)
        return F.conv2d(x, k_full, padding = 1)
        
        
class SteerableOctaveNet(BaseNet):
    def __init__(self, numBlocks, symmDepth, deltaDepth, kxyDepth, kdDepth, numPools = 3, initDepth = 3, numClasses = 10):
        super().__init__()
        self.netName = "Steerable Octave Net"
        self.numBlocks = numBlocks
        blockSize = symmDepth + 2 * deltaDepth + kxyDepth + kdDepth
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = initDepth
        self.initConv = SteerableImageToOctave(size, symmDepth, deltaDepth, kxyDepth, kdDepth)
        size += blockSize
        
        for layer in range(numPools):
            for i in range(numBlocks):
                self.convs[layer].append(SteerableOctaveToOctave(size, symmDepth, deltaDepth, kxyDepth, kdDepth))
                size += blockSize
        
        self.fc = nn.Linear(size, numClasses)
        
    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y, num = 8)
        for layer in range(self.numPools):
            for i in range(self.numBlocks):
                y = self.convs[layer][i](x)
                x = catReluMultipleMultiple(x, y, num = 8)
            x = avgPool2d(x, kSize = self.poolingSizes[layer])
        
        x = x.view(*x.shape[:2])
        x = averageMultiple(x, num = 8)
        x = self.fc(x)
        return x  
        
        
        
        
    
    
def maxminSingle(x):
    return torch.cat((F.relu(x), F.relu(-x)), dim = 1)
    
def maxminMultiple(x):
    _, depth, _, _ = x.shape
    b = x[:, :int(depth/2) ,:,:]
    c = x[:,  int(depth/2):,:,:]
    
    return torch.cat((F.relu(b),F.relu(-c)), dim = 1)
    
class SteerableImageToPair(nn.Module):
    def __init__(self, inpDepth, symmDepth, asymmDepth):
        super().__init__()
        
        self.symmDepth = symmDepth
        
        self.a = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.b = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.c = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.d = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.e = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.f = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        
        
        self.g = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.h = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.i = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.zeros = Variable(torch.zeros(asymmDepth, inpDepth, 3), requires_grad=False)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
        
    def forward(self, x):
        k1 = torch.stack((self.a, self.b, self.c), dim=2)
        k2 = torch.stack((self.d, self.e, self.f), dim=2)
        kernel_s = torch.stack((k1, k2, k1), dim = 3)
        
        k1 = torch.stack((self.g, self.h, self.i), dim=2)
        kernel_a = torch.stack((k1, self.zeros, -k1), dim = 3)
        
        kernel = torch.cat((kernel_s, kernel_a), dim = 0)
        y = F.conv2d(x, kernel, padding = 1)
        
        return torch.cat((y, y[:, :self.symmDepth], -y[:, self.symmDepth:]), dim = 1)
    
    
class SteerablePairToPair(nn.Module):
    def __init__(self, inpDepth, symmDepth, asymmDepth):
        super().__init__()
        self.a1 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.b1 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.c1 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.d1 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.e1 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.f1 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        
        self.a2 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.b2 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.c2 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.d2 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.e2 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        self.f2 = Parameter(torch.randn(symmDepth, inpDepth)* 0.5 / inpDepth)
        
        
        self.g1 = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.h1 = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.i1 = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        
        self.g2 = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.h2 = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        self.i2 = Parameter(torch.randn(asymmDepth, inpDepth)* 0.5 / inpDepth)
        
        self.zeros = Variable(torch.zeros(asymmDepth, inpDepth, 3), requires_grad=False)
        
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
        
    def forward(self, x):
        k1 = torch.stack((self.a1, self.b1, self.c1), dim=2)
        k2 = torch.stack((self.d1, self.e1, self.f1), dim=2)
        kernel_s1 = torch.stack((k1, k2, k1), dim = 3)
        
        k1 = torch.stack((self.a2, self.b2, self.c2), dim=2)
        k2 = torch.stack((self.d2, self.e2, self.f2), dim=2)
        kernel_s2 = torch.stack((k1, k2, k1), dim = 3)
        
        k1 = torch.stack((self.g1, self.h1, self.i1), dim=2)
        kernel_a1 = torch.stack((k1, self.zeros, -k1), dim = 3)
        
        k1 = torch.stack((self.g2, self.h2, self.i2), dim=2)
        kernel_a2 = torch.stack((k1, self.zeros, -k1), dim = 3)
        
        us1 = torch.cat((kernel_s1, kernel_s2), dim = 1)
        ua1 = torch.cat((kernel_a1, kernel_a2), dim = 1)
        us2 = torch.cat((kernel_s2, kernel_s1), dim = 1)
        ua2 = torch.cat((kernel_a2, kernel_a1), dim = 1)
        
        u = torch.cat((us1, ua1, us2, -ua2), dim = 0)
        
        y = F.conv2d(x, u, padding = 1)
        
        return y
    

class SteerablePairNet(BaseNet):
    def __init__(self, numBlocks, symmBlockSize, asymmBlockSize, numPools = 3, initDepth = 3, numClasses = 10):
        super().__init__()
        self.netName = "Steerable Pair Net"
        self.numBlocks = numBlocks
        blockSize = symmBlockSize + asymmBlockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.pools = nn.ModuleList()
        for i in range(numPools-1):
            self.pools.append(avgPool2d(x, kSize = self.poolingSizes[layer]))
        self.pools.append(nn.AdaptiveAvgPool2d(1))        
        
        size = initDepth
        self.initConv = SteerableImageToPair(size, symmBlockSize, asymmBlockSize)
        size += blockSize
        
        for layer in range(numPools):
            for i in range(numBlocks):
                self.convs[layer].append(SteerablePairToPair(size, symmBlockSize, asymmBlockSize))
                size += blockSize
        
        self.fc = nn.Linear(size, numClasses)
        
    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y, num = 2)
        for layer in range(self.numPools):
            for i in range(self.numBlocks):
                y = self.convs[layer][i](x)
                x = catReluMultipleMultiple(x, y, num = 2)
            x = self.pools[layer](x)
        
        x = x.view(*x.shape[:2])
        x = averageMultiple(x, num = 2)
        x = self.fc(x)
        return x



    
class SteerableImageToAssymetricPair(nn.Module):
    def __init__(self, inpDepth, outDepth):
        super().__init__()
        self.a = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.b = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.c = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.zeros = Variable(torch.zeros(outDepth, inpDepth, 3), requires_grad=False)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            
    def forward(self, x):
        k1 = torch.stack((self.a, self.b, self.c), dim=2)
        kernel = torch.stack((k1, self.zeros, -k1), dim = 3)
        y = F.conv2d(x, kernel, padding = 1)
        return y
        #y = maxminSingle(y)
        #x = catSingleMultiple(x, y, num = 2)
        #return x
    
class SteerablePairToAssymetricPair(nn.Module):
    def __init__(self, inpDepth, outDepth):
        super().__init__()
        self.a1 = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.b1 = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.c1 = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        
        self.a2 = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.b2 = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        self.c2 = Parameter(torch.randn(outDepth, inpDepth)* 0.5 / inpDepth)
        
        self.zeros = Variable(torch.zeros(outDepth, inpDepth, 3), requires_grad=False)
        
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
        
    def forward(self, x):
        k1 = torch.stack((self.a1, self.b1, self.c1), dim=2)
        kernel1 = torch.stack((k1, self.zeros, -k1), dim = 3)
        
        k2 = torch.stack((self.a2, self.b2, self.c2), dim=2)
        kernel2 = torch.stack((k2, self.zeros, -k2), dim = 3)
        
        u1 = torch.cat((kernel1, kernel2), dim = 1)
        u2 = torch.cat((kernel2, kernel1), dim = 1)
        
        u = torch.cat((u1, u2), dim = 0)
        
        y = F.conv2d(x, u, padding = 1)
        
        return y
        
        #y = maxminMultiple(y)
        
        #return catMultipleMultiple(x, y, num = 2)
        
        
class SteerableAssymetricPairNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3):
        super().__init__()
        self.netName = "Steerable Pair Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        
        size = 3
        self.initConv = SteerableImageToPair(size, blockSize)
        size += blockSize
        
        for layer in range(numPools):
            for i in range(numBlocks):
                self.convs[layer].append(SteerablePairToPair(size, blockSize))
                size += blockSize
        
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        y = self.initConv(x)
        y = maxminSingle(y)
        x = catSingleMultiple(x, y, num = 2)
        for layer in range(self.numPools):
            for i in range(self.numBlocks):
                y = self.convs[layer][i](x)
                y = maxminMultiple(y)
                x = catMultipleMultiple(x, y, num = 2)
            x = avgPool2d(x, kSize = self.poolingSizes[layer])
        
        x = x.view(*x.shape[:2])
        x = averageMultiple(x, num = 2)
        x = self.fc(x)
        return x
    
        
        
        
    
    
    
class UsualDenseNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3, initDepth = 3, numClasses = 10):
        super().__init__()
        self.netName = "Usual Dense Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
        
        self.pools = nn.ModuleList()
        for pool in range(numPools-1):
            self.pools.append(nn.AvgPool2d(2))
        self.pools.append(nn.AdaptiveAvgPool2d(1))
        
        size = initDepth
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
        self.fc = nn.Linear(size, numClasses)
        
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x
    
    def freezeConvLayers(self):
        for param in self.convs.parameters():
            param.requires_grad = False
            
            
class Conv1dPool(nn.Module):
    def __init__(self, initDepth, reductionFactor = 0.5):
        super().__init__()
        self.outDepth = int(initDepth * reductionFactor)
        self.conv1d = nn.Conv2d(initDepth, self.outDepth, 1)
        self.outDepth = int(initDepth * reductionFactor)
    
    def forward(self, x):
        return F.avg_pool2d(self.conv1d(x), 2)
    
    def getOutDepth(self):
        return self.outDepth
    
class DenseNetImproved(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3, initDepth = 3, numClasses = 10):
        super().__init__()
        self.netName = "Dense Net Improved"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs   = nn.ModuleList()
        self.convs1d = nn.ModuleList()
        self.pools   = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            self.convs1d.append(nn.ModuleList())
        
        self.pools = nn.ModuleList()
        
        size = initDepth
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs1d[layer].append(nn.Conv2d(size, blockSize, 1))
                self.convs[layer].append(nn.Conv2d(blockSize, blockSize, 3, padding = 1))       
                size += blockSize
            if (layer != numPools - 1):
                self.pools.append(Conv1dPool(size))
                size = self.pools[-1].getOutDepth()
        self.pools.append(nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(size, numClasses)
        
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](self.convs1d[layer][block](x)))), dim = 1)
            x = self.pools[layer](x)
        x = x.view(x.shape[:2])
        x = self.fc(x)
        return x
    
    def freezeConvLayers(self):
        for param in self.convs.parameters():
            param.requires_grad = False
            
            
class SoftMaxNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3):
        super().__init__()
        self.netName = "Usual Dense Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
        self.fc = nn.Linear(size, 10)
        
        self.soft2d = nn.Softmax2d()
        
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, self.soft2d(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        x = self.fc(x.squeeze())
        return x
      
    
class CrossEntropyLogLoss(nn.Module):
    def __init__(self, reduction = "mean"):
        super().__init__()
        self.nlll = nn.NLLLoss(reduction = reduction)
        
    def forward(self, inp, labels):
        return self.nlll(torch.log(inp + 0.0001), labels)
        
class MarginNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3):
        super().__init__()
        self.netName = "Max Vote Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        #self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            if (layer == numPools - 1):
                self.pools.append(MinProbPool(size))
            else:
                self.pools.append(nn.AvgPool2d(2))
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        #x = self.fc(x.squeeze())
        return x
    
    def getLoss(self, reduction = "mean"):
        return CrossEntropyLogLoss(reduction = reduction)
            
          
            
    
class MaxVoteNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3):
        super().__init__()
        self.netName = "Max Vote Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        #self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            if (layer == numPools - 1):
                self.pools.append(MaxProbPool(size))
            else:
                self.pools.append(nn.AvgPool2d(2))
        #self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        #x = self.fc(x.squeeze())
        return x
    
    #def getLoss(self, reduction = "mean"):
    #    return nn.BCELoss(reduction = reduction)
    
    #def convertLabels(self, labels):
    #    mb = labels.shape[0]
    #    outDepth = 10
    #    y = torch.zeros(mb, outDepth)
    #    if torch.cuda.is_available():
    #        y = y.cuda()
    #    z = y.scatter_(1, labels.view(-1,1), 1)
    #    return z
    
    
    
class MinProbPool(nn.Module):
    def __init__(self, inputDepth, outputDepth = 10):
        super().__init__()
        self.conv1d = nn.Conv2d(inputDepth, outputDepth, 1)
        #self.fc = nn.Linear(inputDepth, outputDepth)
        self.outputDepth = outputDepth
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.soft2d = nn.Softmax2d()
        
        
    def forward(self, x):
        if (self.training):
            x = self.conv1d(x)
            x = self.soft2d(x)
            x = - self.max_pool(-x)
            x = x.squeeze()
            return x
        
            #x = x.permute(0,2,3,1)
            #x = self.fc(x)
            #x = x.softmax(-1)
            #x = x.permute(0, 3, 1, 2)
            #x = - self.max_pool(-x)
            #x = x.squeeze()
            #return x
        else:
            x = self.avg_pool(x)
            x = self.conv1d(x)
            x = x.squeeze()
            x = x.softmax(-1)
            return x
            
            
        
        
        
class MaxProbPool(nn.Module):
    def __init__(self, inputDepth, outputDepth = 10):
        super().__init__()
        self.fc = nn.Linear(inputDepth, outputDepth)
        self.outputDepth = outputDepth
        self.pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.fc(x)
        x = x.softmax(-1)
        x = x.permute(0, 3, 1, 2)
        x = self.pool(x)
        x = x.squeeze()
        return x
        

         
            
class PointToVectSobel(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.conv = nn.Conv2d(inputDepth, outputDepth, 1)
        self.sobel = Sobel(outputDepth)
        
    def forward(self, x):
        vx, vy = self.sobel(self.conv(x))
        return torch.cat((vx, vy), dim = 1)
    
    
    
class VectToVectConvEfficient(nn.Module):
    def __init__(self, inputDepth, outDepth):
        super().__init__()
        
        self.a = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.b = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.c = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.d = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.e = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.zeros = Variable(torch.zeros(outDepth, inputDepth), requires_grad=False)
        if torch.cuda.is_available():
            self.zeros = self.zeros.cuda()
            
    def forward(self, x):
        k1 = torch.stack((self.a, self.b, self.a), dim=2)
        k2 = torch.stack((self.c, self.d, self.c), dim=2)
        kernel_xx = torch.stack((k1, k2, k1), dim=3)
        
        k1 = torch.stack((self.e, self.zeros, -self.e), dim=2)
        k2 = torch.stack((self.zeros, self.zeros, self.zeros), dim=2)
        kernel_xy = torch.stack((k1, k2, -k1), dim=3)
        
        kernel_yy = kernel_xx.transpose(-1, -2)
        
        kernel_x = torch.cat((kernel_xx, kernel_xy), dim = 1)
        kernel_y = torch.cat((kernel_xy, kernel_yy), dim = 1)
        
        kernel = torch.cat((kernel_x, kernel_y), dim = 0)
        
        return F.conv2d(x, kernel, padding = 1)
        
        
class DerotateVectorsEfficient(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        _, depth_2 = x.shape
        depth = int(depth_2/2)
        
        vx = x[:, :depth]
        vy = x[:, depth:]
        
        vx_avg = torch.mean(vx, dim = 1, keepdim = True)
        vy_avg = torch.mean(vy, dim = 1, keepdim = True)
        
        l = torch.sqrt(vx_avg * vx_avg + vy_avg * vy_avg + 0.01)
        
        vx_avg = vx_avg / l
        vy_avg = vy_avg / l
        
        vx_res =  vx * vx_avg + vy * vy_avg
        vy_res = -vx * vy_avg + vy * vx_avg
        
        return torch.cat((vx_res, vy_res), dim = 1)

        
        
class GradientNetwork(BaseNet):
    def __init__(self, numBlocks, blockSize, initConvDepth, numPools = 3, derotation = False):
        super().__init__()
        self.numBlocks = numBlocks
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        self.initConv = PointToVectSobel(3, initConvDepth)
        
        size = initConvDepth
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(VectToVectConvEfficient(size, blockSize))               
                size += blockSize
            self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
        if (derotation):
            self.derotate = DerotateVectorsEfficient()
        else:
            self.derotate = Identity()
        self.fc = nn.Linear(2 * size, 10)
        
    def forward(self, x):
        x = self.initConv(x)
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = catMultipleMultiple(x, squash(self.convs[layer][block](x)))
            x = self.pools[layer](x)
        x = x.view(x.shape[:2])
        x = self.derotate(x)
        x = self.fc(x)
        return x
        
        
        
class SequentialLinear(nn.Module):
    def __init__(self, inSize, numBlocks, blockSize, outSize = 10):
        super().__init__()
        self.numBlocks = numBlocks
        self.linears = nn.ModuleList()
        size = inSize
        for block in range(numBlocks):
            self.linears.append(nn.Linear(size, blockSize))
            size += blockSize
        self.fc = nn.Linear(size, outSize)
    
    def forward(self, x):
        for block in range(self.numBlocks):
            x = torch.cat((x, torch.relu(self.linears[block](x))), dim = 1)
        return self.fc(x)
        
        
        
class ImageToPairConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.kernel = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        kernel_t = self.kernel.flip(3)
        
        left = F.conv2d(x, self.kernel, padding = 1)
        right = F.conv2d(x, kernel_t, padding = 1)
        
        return left, right
    
    
class PairToPairConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.kernel_a = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.kernel_b = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, left, right):
        kernel_a_t = self.kernel_a.flip(3)
        kernel_b_t = self.kernel_b.flip(3)
        
        left_res = F.conv2d(left, self.kernel_a, padding = 1) + F.conv2d(right, self.kernel_b, padding = 1)
        right_res = F.conv2d(left, kernel_b_t, padding = 1) + F.conv2d(right, kernel_a_t, padding = 1)
        
        return left_res, right_res
    

def rotate_c(x):
    #rotates 4d tensor 90 degrees clockwisely
    return x.transpose(-2, -1).flip(-1)

def rotate_c_(*args):
    return [rotate_c(arg) for arg in args]

def rotate_cc(x):
    #rotates 4d tensor 90 degrees clockwisely
    return x.transpose(-2, -1).flip(-2)

def rotate_cc_(*args):
    return [rotate_cc(arg) for arg in args]

def rotate4(k):
    k_r = rotate_c(k)
    return [k, k_r, rotate_c(k_r), rotate_cc(k)] 

def rotate4_cc(k):
    k_r = rotate_cc(k)
    return [k, k_r, rotate_cc(k_r), rotate_c(k)] 
    
    
def flip(k):
    return k.flip(3)

def flip_(*args):
    return [arg.flip(3) for arg in args]

def conv(x, kernel):
    return F.conv2d(x, kernel, padding = 1)

def avgPool2d(x, kSize = 2):
    return F.avg_pool2d(x, kSize)

def avgPool2d_(*args, kSize = 2):
    return [F.avg_pool2d(arg, kSize) for arg in args]

def cat_(x, *args):
    return [torch.cat((x, arg), dim = 1) for arg in args]

def catRelu_(x, *args):
    return [torch.cat((x, F.relu(arg)), dim = 1) for arg in args]

def catReluList(l1, l2):
    return [torch.cat((l1[i], F.relu(l2[i])), dim = 1) for i in range(len(l1))]

def squeeze_(*args):
    return [arg.squeeze() for arg in args]

def averageMultiple(x, num = 8):
    mb, num_depth = x.shape[:2]
    x = x.view(mb, num, int(num_depth / num), *x.shape[2:]).mean(dim = 1)
    return x

def maxMultiple(x, num = 8):
    mb, num_depth = x.shape[:2]
    x = x.view(mb, num, int(num_depth / num), *x.shape[2:]).max(dim = 1)[0]
    return x

def sumOfSquaresMultiple(x, num = 8):
    mb, num_depth = x.shape[:2]
    x = x.view(mb, num, int(num_depth / num), *x.shape[2:])
    x = x * x
    x = x.mean(dim = 1)
    return x

def convList(x, kernels):
    #kernels is a tuple of kernels of the same shape
    k = torch.cat(kernels, dim = 0)
    res = F.conv2d(x, k, padding = 1)
    return torch.split(res, kernels[0].shape[0], dim = 1)

def sumConvList(inputs, kernels):
    inp = torch.cat(inputs, dim = 1)
    k = torch.cat(kernels, dim = 1)
    return F.conv2d(inp, k, padding = 1)

#def catReluSingleMultiple(x, y, num = 8):
#    y_splitted = torch.split(F.relu(y), y.shape[1] / num, dim = 1)
#    y_catted = cat_(x, *ySplitted)
#    return torch.cat(y_catted, dim = 1)

def catReluSingleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)
    
    y = F.relu(y).view(mb, num, int(num_blockSize / num), rows, cols)
    x = x.unsqueeze(1).expand(-1, num, -1, -1, -1)
    
    res = torch.cat((x,y), dim = 2).view(mb, -1, rows, cols)
    return res

def catSingleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)
    
    y = y.view(mb, num, int(num_blockSize / num), rows, cols)
    x = x.unsqueeze(1).expand(-1, num, -1, -1, -1)
    
    res = torch.cat((x,y), dim = 2).view(mb, -1, rows, cols)
    return res

def catMultipleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)
    
    x = x.view(mb, num, -1, rows, cols)
    y = y.view(mb, num, -1, rows, cols)
    
    res = torch.cat((x,y), dim = 2).view(mb, -1, rows, cols).view(mb, -1, rows, cols)
    return res

def squash(x):
    mb, _, rows, cols = x.shape
    x = x.view(mb, 2, -1, rows, cols)
    l = torch.sqrt((x * x).sum(dim = 1, keepdim = True) + 1)
    return (x / l).view(mb, -1, rows, cols)

def catReluMultipleMultiple(x, y, num = 8):
    mb, num_blockSize, rows, cols = list(y.shape)
    
    x = x.view(mb, num, -1, rows, cols)
    y = F.relu(y).view(mb, num, int(num_blockSize / num), rows, cols)
    
    res = torch.cat((x,y), dim = 2).view(mb, -1, rows, cols)
    return res
    


class ImageToQuadConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.kernel = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        kernel_r = rotate_c(self.kernel)
        kernel_rr = rotate_c(kernel_r)
        kernel_rrr = rotate_cc(self.kernel)
        
        top = F.conv2d(x, self.kernel, padding = 1)
        right = F.conv2d(x, kernel_r, padding = 1)
        bottom = F.conv2d(x, kernel_rr, padding = 1)
        left = F.conv2d(x, kernel_rrr, padding = 1)
        
        return top, right, bottom, left
    
    
class QuadToQuadConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k_a = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_b = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_c = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_d = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, top, right, bottom, left):
        k_a_r,   k_b_r,   k_c_r,   k_d_r   = rotate_c_ (self.k_a, self.k_b, self.k_c, self.k_d)
        k_a_rr,  k_b_rr,  k_c_rr,  k_d_rr  = rotate_c_ (k_a_r,    k_b_r,    k_c_r,    k_d_r)
        k_a_rrr, k_b_rrr, k_c_rrr, k_d_rrr = rotate_cc_(self.k_a, self.k_b, self.k_c, self.k_d)
                
        top_res    = conv(top,    self.k_a) + conv(right,  self.k_b) + conv(bottom, self.k_c) + conv(left,   self.k_d)
        right_res  = conv(right,  k_a_r)    + conv(bottom, k_b_r)    + conv(left,   k_c_r)    + conv(top,    k_d_r)
        bottom_res = conv(bottom, k_a_rr)   + conv(left,   k_b_rr)   + conv(top,    k_c_rr)   + conv(right,  k_d_rr)
        left_res   = conv(left,   k_a_rrr)  + conv(top,    k_b_rrr)  + conv(right,  k_c_rrr)  + conv(bottom, k_d_rrr)
        
        return top_res, right_res, bottom_res, left_res
    
    
class ImageToQuadEfficientConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.kernel = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        k, k_r, k_rr, k_rrr = rotate4(self.kernel)
        kernel = torch.cat((k, k_r, k_rr, k_rrr), dim = 0)
        return F.conv2d(x, kernel, padding = 1)
    
class QuadToQuadEfficientConv1d(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k_a = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_b = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_c = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_d = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        
    def forward(self, x):
        k_a, k_b, k_c, k_d = self.k_a, self.k_b, self.k_c, self.k_d
        
        k1 = torch.cat((k_a, k_b, k_c, k_d), dim = 1)
        k2 = torch.cat((k_d, k_a, k_b, k_c), dim = 1)
        k3 = torch.cat((k_c, k_d, k_a, k_b), dim = 1)
        k4 = torch.cat((k_b, k_c, k_d, k_a), dim = 1)
        
        kernel = torch.cat((k1, k2, k3, k4), dim = 0)
        
        return F.conv2d(x, kernel)
    
class QuadToQuadEfficientConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k_a = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_b = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_c = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_d = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        k_a, k_b, k_c, k_d = self.k_a, self.k_b, self.k_c, self.k_d
        k_a_r,   k_b_r,   k_c_r,   k_d_r   = rotate_c_ (self.k_a, self.k_b, self.k_c, self.k_d)
        k_a_rr,  k_b_rr,  k_c_rr,  k_d_rr  = rotate_c_ (k_a_r,    k_b_r,    k_c_r,    k_d_r)
        k_a_rrr, k_b_rrr, k_c_rrr, k_d_rrr = rotate_cc_(self.k_a, self.k_b, self.k_c, self.k_d)
        
        k1 = torch.cat((k_a,     k_b,     k_c,     k_d)    , dim = 1)
        k2 = torch.cat((k_d_r,   k_a_r,   k_b_r,   k_c_r)  , dim = 1)
        k3 = torch.cat((k_c_rr,  k_d_rr,  k_a_rr,  k_b_rr) , dim = 1)
        k4 = torch.cat((k_b_rrr, k_c_rrr, k_d_rrr, k_a_rrr), dim = 1)
        
        kernel = torch.cat((k1, k2, k3, k4), dim = 0)
        
        return F.conv2d(x, kernel, padding = 1)


class ImageToOctaveConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        k_f = flip(self.k)
        k,   k_r,   k_rr,   k_rrr   = rotate4(self.k)
        k_f, k_r_f, k_rr_f, k_rrr_f = rotate4(k_f)
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = convList(x, (k, k_f, k_r, k_r_f, k_rr, k_rr_f, k_rrr, k_rrr_f))
        return t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f
    
    
class ImageToOctaveEfficientConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        k_f = flip(self.k)
        k,   k_r,   k_rr,   k_rrr   = rotate4(self.k)
        k_f, k_r_f, k_rr_f, k_rrr_f = rotate4(k_f)
        k = torch.cat((k, k_f, k_r, k_r_f, k_rr, k_rr_f, k_rrr, k_rrr_f), dim = 0)
        return F.conv2d(x, k, padding = 1)
    
    
    
class OctaveToOctaveConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k_c_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_c_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
        self.k_r_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_r_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
        self.k_b_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_b_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
        self.k_l_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_l_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f):
        
        k_c_0, k_c_0_r, k_c_0_rr, k_c_0_rrr = rotate4(self.k_c_0)
        k_r_0, k_r_0_r, k_r_0_rr, k_r_0_rrr = rotate4(self.k_r_0)
        k_b_0, k_b_0_r, k_b_0_rr, k_b_0_rrr = rotate4(self.k_b_0)
        k_l_0, k_l_0_r, k_l_0_rr, k_l_0_rrr = rotate4(self.k_l_0)
        
        k_c_1, k_c_1_r, k_c_1_rr, k_c_1_rrr = rotate4(self.k_c_1)
        k_r_1, k_r_1_r, k_r_1_rr, k_r_1_rrr = rotate4(self.k_r_1)
        k_b_1, k_b_1_r, k_b_1_rr, k_b_1_rrr = rotate4(self.k_b_1)
        k_l_1, k_l_1_r, k_l_1_rr, k_l_1_rrr = rotate4(self.k_l_1)
        
        k_c_0_f, k_c_0_r_f, k_c_0_rr_f, k_c_0_rrr_f = flip_(k_c_0, k_c_0_r, k_c_0_rr, k_c_0_rrr)
        k_r_0_f, k_r_0_r_f, k_r_0_rr_f, k_r_0_rrr_f = flip_(k_r_0, k_r_0_r, k_r_0_rr, k_r_0_rrr)
        k_b_0_f, k_b_0_r_f, k_b_0_rr_f, k_b_0_rrr_f = flip_(k_b_0, k_b_0_r, k_b_0_rr, k_b_0_rrr)
        k_l_0_f, k_l_0_r_f, k_l_0_rr_f, k_l_0_rrr_f = flip_(k_l_0, k_l_0_r, k_l_0_rr, k_l_0_rrr)
        
        k_c_1_f, k_c_1_r_f, k_c_1_rr_f, k_c_1_rrr_f = flip_(k_c_1, k_c_1_r, k_c_1_rr, k_c_1_rrr)
        k_r_1_f, k_r_1_r_f, k_r_1_rr_f, k_r_1_rrr_f = flip_(k_r_1, k_r_1_r, k_r_1_rr, k_r_1_rrr)
        k_b_1_f, k_b_1_r_f, k_b_1_rr_f, k_b_1_rrr_f = flip_(k_b_1, k_b_1_r, k_b_1_rr, k_b_1_rrr)
        k_l_1_f, k_l_1_r_f, k_l_1_rr_f, k_l_1_rrr_f = flip_(k_l_1, k_l_1_r, k_l_1_rr, k_l_1_rrr)
        
        
        t_n_res = sumConvList((t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f), (k_c_0,   k_c_1,   k_r_0,   k_r_1,   k_b_0,   k_b_1,   k_l_0,   k_l_1))
        t_f_res = sumConvList((t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f), (k_c_1_f, k_c_0_f, k_l_1_f, k_l_0_f, k_b_1_f, k_b_0_f, k_r_1_f, k_r_0_f))
        
        r_n_res = sumConvList((r_n, r_f, b_n, b_f, l_n, l_f, t_n, t_f), (k_c_0_r,   k_c_1_r,   k_r_0_r,   k_r_1_r,   k_b_0_r,   k_b_1_r,   k_l_0_r,   k_l_1_r))
        l_f_res = sumConvList((r_n, r_f, b_n, b_f, l_n, l_f, t_n, t_f), (k_b_1_r_f, k_b_0_r_f, k_r_1_r_f, k_r_0_r_f, k_c_1_r_f, k_c_0_r_f, k_l_1_r_f, k_l_0_r_f))
	        
        b_n_res = sumConvList((b_n, b_f, l_n, l_f, t_n, t_f, r_n, r_f), (k_c_0_rr,   k_c_1_rr,   k_r_0_rr,   k_r_1_rr,   k_b_0_rr,   k_b_1_rr,   k_l_0_rr,   k_l_1_rr))
        b_f_res = sumConvList((b_n, b_f, l_n, l_f, t_n, t_f, r_n, r_f), (k_c_1_rr_f, k_c_0_rr_f, k_l_1_rr_f, k_l_0_rr_f, k_b_1_rr_f, k_b_0_rr_f, k_r_1_rr_f, k_r_0_rr_f))
        
        l_n_res = sumConvList((l_n, l_f, t_n, t_f, r_n, r_f, b_n, b_f), (k_c_0_rrr,   k_c_1_rrr,   k_r_0_rrr,   k_r_1_rrr,   k_b_0_rrr,   k_b_1_rrr,   k_l_0_rrr,   k_l_1_rrr))
        r_f_res = sumConvList((l_n, l_f, t_n, t_f, r_n, r_f, b_n, b_f), (k_b_1_rrr_f, k_b_0_rrr_f, k_r_1_rrr_f, k_r_0_rrr_f, k_c_1_rrr_f, k_c_0_rrr_f, k_l_1_rrr_f, k_l_0_rrr_f))
        
        return t_n_res, t_f_res, r_n_res, r_f_res, b_n_res, b_f_res, l_n_res, l_f_res
    
    
class OctaveToOctaveEfficientConv1d(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k_c_0 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_c_1 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        
        self.k_r_0 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_r_1 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        
        self.k_b_0 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_b_1 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        
        self.k_l_0 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        self.k_l_1 = Parameter(torch.randn(outputDepth, inputDepth, 1, 1) * 0.5 / inputDepth)
        
    def forward(self, x):
        k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1 = self.k_c_0, self.k_c_1, self.k_r_0, self.k_r_1, self.k_b_0, self.k_b_1, self.k_l_0, self.k_l_1
        
        k_t_n = torch.cat((k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1), dim = 1)
        k_t_f = torch.cat((k_c_1, k_c_0, k_l_1, k_l_0, k_b_1, k_b_0, k_r_1, k_r_0), dim = 1)
        k_r_n = torch.cat((k_l_0, k_l_1, k_c_0, k_c_1, k_r_0, k_r_1, k_b_0, k_b_1), dim = 1)
        k_r_f = torch.cat((k_r_1, k_r_0, k_c_1, k_c_0, k_l_1, k_l_0, k_b_1, k_b_0), dim = 1)
        k_b_n = torch.cat((k_b_0, k_b_1, k_l_0, k_l_1, k_c_0, k_c_1, k_r_0, k_r_1), dim = 1)
        k_b_f = torch.cat((k_b_1, k_b_0, k_r_1, k_r_0, k_c_1, k_c_0, k_l_1, k_l_0), dim = 1)
        k_l_n = torch.cat((k_r_0, k_r_1, k_b_0, k_b_1, k_l_0, k_l_1, k_c_0, k_c_1), dim = 1)
        k_l_f = torch.cat((k_l_1, k_l_0, k_b_1, k_b_0, k_r_1, k_r_0, k_c_1, k_c_0), dim = 1)
        k = torch.cat((k_t_n, k_t_f, k_r_n, k_r_f, k_b_n, k_b_f, k_l_n, k_l_f), dim = 0)
        return F.conv2d(x, k)
    
    
class OctaveToOctaveEfficientConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        self.k_c_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_c_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
        self.k_r_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_r_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
        self.k_b_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_b_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
        self.k_l_0 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        self.k_l_1 = Parameter(torch.randn(outputDepth, inputDepth, 3, 3) * 0.5 / inputDepth)
        
    def forward(self, x):
        k_c_0, k_c_0_r, k_c_0_rr, k_c_0_rrr = rotate4(self.k_c_0)
        k_r_0, k_r_0_r, k_r_0_rr, k_r_0_rrr = rotate4(self.k_r_0)
        k_b_0, k_b_0_r, k_b_0_rr, k_b_0_rrr = rotate4(self.k_b_0)
        k_l_0, k_l_0_r, k_l_0_rr, k_l_0_rrr = rotate4(self.k_l_0)
        
        k_c_1, k_c_1_r, k_c_1_rr, k_c_1_rrr = rotate4(self.k_c_1)
        k_r_1, k_r_1_r, k_r_1_rr, k_r_1_rrr = rotate4(self.k_r_1)
        k_b_1, k_b_1_r, k_b_1_rr, k_b_1_rrr = rotate4(self.k_b_1)
        k_l_1, k_l_1_r, k_l_1_rr, k_l_1_rrr = rotate4(self.k_l_1)
        
        k_c_0_f, k_c_0_r_f, k_c_0_rr_f, k_c_0_rrr_f = flip_(k_c_0, k_c_0_r, k_c_0_rr, k_c_0_rrr)
        k_r_0_f, k_r_0_r_f, k_r_0_rr_f, k_r_0_rrr_f = flip_(k_r_0, k_r_0_r, k_r_0_rr, k_r_0_rrr)
        k_b_0_f, k_b_0_r_f, k_b_0_rr_f, k_b_0_rrr_f = flip_(k_b_0, k_b_0_r, k_b_0_rr, k_b_0_rrr)
        k_l_0_f, k_l_0_r_f, k_l_0_rr_f, k_l_0_rrr_f = flip_(k_l_0, k_l_0_r, k_l_0_rr, k_l_0_rrr)
        
        k_c_1_f, k_c_1_r_f, k_c_1_rr_f, k_c_1_rrr_f = flip_(k_c_1, k_c_1_r, k_c_1_rr, k_c_1_rrr)
        k_r_1_f, k_r_1_r_f, k_r_1_rr_f, k_r_1_rrr_f = flip_(k_r_1, k_r_1_r, k_r_1_rr, k_r_1_rrr)
        k_b_1_f, k_b_1_r_f, k_b_1_rr_f, k_b_1_rrr_f = flip_(k_b_1, k_b_1_r, k_b_1_rr, k_b_1_rrr)
        k_l_1_f, k_l_1_r_f, k_l_1_rr_f, k_l_1_rrr_f = flip_(k_l_1, k_l_1_r, k_l_1_rr, k_l_1_rrr)
        
        k_t_n = torch.cat((k_c_0,   k_c_1,   k_r_0,   k_r_1,   k_b_0,   k_b_1,   k_l_0,   k_l_1), dim = 1)
        k_t_f = torch.cat((k_c_1_f, k_c_0_f, k_l_1_f, k_l_0_f, k_b_1_f, k_b_0_f, k_r_1_f, k_r_0_f), dim = 1)
        k_r_n = torch.cat((k_l_0_r, k_l_1_r, k_c_0_r, k_c_1_r, k_r_0_r, k_r_1_r, k_b_0_r, k_b_1_r), dim = 1)
        k_r_f = torch.cat((k_r_1_rrr_f, k_r_0_rrr_f, k_c_1_rrr_f, k_c_0_rrr_f, k_l_1_rrr_f, k_l_0_rrr_f, k_b_1_rrr_f, k_b_0_rrr_f), dim = 1)
        k_b_n = torch.cat((k_b_0_rr, k_b_1_rr, k_l_0_rr, k_l_1_rr, k_c_0_rr, k_c_1_rr, k_r_0_rr, k_r_1_rr), dim = 1)
        k_b_f = torch.cat((k_b_1_rr_f, k_b_0_rr_f, k_r_1_rr_f, k_r_0_rr_f, k_c_1_rr_f, k_c_0_rr_f, k_l_1_rr_f, k_l_0_rr_f), dim = 1)
        k_l_n = torch.cat((k_r_0_rrr,   k_r_1_rrr,   k_b_0_rrr,   k_b_1_rrr,   k_l_0_rrr,   k_l_1_rrr, k_c_0_rrr,   k_c_1_rrr), dim = 1)
        k_l_f = torch.cat((k_l_1_r_f, k_l_0_r_f, k_b_1_r_f, k_b_0_r_f, k_r_1_r_f, k_r_0_r_f, k_c_1_r_f, k_c_0_r_f), dim = 1)
        
        k = torch.cat((k_t_n, k_t_f, k_r_n, k_r_f, k_b_n, k_b_f, k_l_n, k_l_f), dim = 0)
        return F.conv2d(x, k, padding = 1)
    
    
class OctaveBatchNorm(nn.Module):
    def __init__(self, depth, num = 8):
        super().__init__()
        self.num = num
        self.depth = depth
        self.bn = nn.BatchNorm2d(num_features = depth)
        
    def forward(self, x):
        mb, num_depth, rows, cols = list(x.shape)
        assert self.num * self.depth == num_depth
        x = x.view(mb, self.num, self.depth, rows, cols)
        x = x.transpose(1, 2).contiguous().view(mb, self.depth, self.num * rows, cols)
        x = self.bn(x)
        x = x.view(mb, self.depth, self.num, rows, cols).transpose(1, 2)
        x = x.contiguous().view(mb, self.num * self.depth, rows, cols)
        return x
    

    
class DualNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Dual Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        self.initConv = ImageToPairConv(size, blockSize)
        size += blockSize
        
        for i in range(numBlocks):
            self.convList1.append(PairToPairConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(PairToPairConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(PairToPairConv(size, blockSize))
            size += blockSize
        
        self.pool1_left = nn.AvgPool2d(2, 2)
        self.pool1_right = nn.AvgPool2d(2, 2)
        self.pool2_left = nn.AvgPool2d(2, 2)
        self.pool2_right = nn.AvgPool2d(2, 2)
        self.pool3_left = nn.AvgPool2d(8, 8)
        self.pool3_right = nn.AvgPool2d(8, 8)
        
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        left, right = self.initConv(x)
        left = torch.cat((x, F.relu(left)), dim = 1)
        right = torch.cat((x, F.relu(right)), dim = 1)
        
        for i in range(self.numBlocks):
            left_new, right_new = self.convList1[i](left, right)
            left = torch.cat((left, F.relu(left_new)), dim = 1)
            right = torch.cat((right, F.relu(right_new)), dim = 1)
            
        left = self.pool1_left(left)
        right = self.pool1_right(right)
        
        for i in range(self.numBlocks):
            left_new, right_new = self.convList2[i](left, right)
            left = torch.cat((left, F.relu(left_new)), dim = 1)
            right = torch.cat((right, F.relu(right_new)), dim = 1)
        
        left = self.pool2_left(left)
        right = self.pool2_right(right)
        
        
        for i in range(self.numBlocks):
            left_new, right_new = self.convList3[i](left, right)
            left = torch.cat((left, F.relu(left_new)), dim = 1)
            right = torch.cat((right, F.relu(right_new)), dim = 1)
        
        left = self.pool3_left(left).squeeze()
        right = self.pool3_right(right).squeeze()
        
        x = (left + right) / 2.0
        x = self.fc(x)
        return x

    
class QuadNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Quad Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        self.initConv = ImageToQuadConv(size, blockSize)
        size += blockSize
        
        for i in range(numBlocks):
            self.convList1.append(QuadToQuadConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(QuadToQuadConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(QuadToQuadConv(size, blockSize))
            size += blockSize
        
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        top, right, bottom, left = self.initConv(x)
        top, right, bottom, left = catRelu_(x, top, right, bottom, left)
        
        for i in range(self.numBlocks):
            top_new, right_new, bottom_new, left_new = self.convList1[i](top, right, bottom, left)
            top, right, bottom, left = catReluList([top, right, bottom, left], [top_new, right_new, bottom_new, left_new])
        
        top, right, bottom, left = avgPool2d_(top, right, bottom, left)
        
        for i in range(self.numBlocks):
            top_new, right_new, bottom_new, left_new = self.convList2[i](top, right, bottom, left)
            top, right, bottom, left = catReluList([top, right, bottom, left], [top_new, right_new, bottom_new, left_new])
        
        top, right, bottom, left = avgPool2d_(top, right, bottom, left)
        
        for i in range(self.numBlocks):
            top_new, right_new, bottom_new, left_new = self.convList3[i](top, right, bottom, left)
            top, right, bottom, left = catReluList([top, right, bottom, left], [top_new, right_new, bottom_new, left_new])
        
        top, right, bottom, left = avgPool2d_(top, right, bottom, left, kSize = 8)
        
        top, right, bottom, left = squeeze_(top, right, bottom, left)
        
        x = (top + right + bottom + left) / 4.0
        x = self.fc(x)
        return x
    
    
class OctaveNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Octave Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        self.initConv = ImageToOctaveConv(size, blockSize)
        size += blockSize
        
        for i in range(numBlocks):
            self.convList1.append(OctaveToOctaveConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(OctaveToOctaveConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(OctaveToOctaveConv(size, blockSize))
            size += blockSize
        
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = self.initConv(x)
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = catRelu_(x, t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
        
        for i in range(self.numBlocks):
            t_n_new, t_f_new, r_n_new, r_f_new, b_n_new, b_f_new, l_n_new, l_f_new = self.convList1[i](t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
            t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = catReluList([t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f], [t_n_new, t_f_new, r_n_new, r_f_new, b_n_new, b_f_new, l_n_new, l_f_new])
        
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = avgPool2d_(t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
        
        for i in range(self.numBlocks):
            t_n_new, t_f_new, r_n_new, r_f_new, b_n_new, b_f_new, l_n_new, l_f_new = self.convList2[i](t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
            t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = catReluList([t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f], [t_n_new, t_f_new, r_n_new, r_f_new, b_n_new, b_f_new, l_n_new, l_f_new])
        
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = avgPool2d_(t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
        
        for i in range(self.numBlocks):
            t_n_new, t_f_new, r_n_new, r_f_new, b_n_new, b_f_new, l_n_new, l_f_new = self.convList3[i](t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
            t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = catReluList([t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f], [t_n_new, t_f_new, r_n_new, r_f_new, b_n_new, b_f_new, l_n_new, l_f_new])
        
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = avgPool2d_(t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f, kSize = 8)
        
        t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f = squeeze_(t_n, t_f, r_n, r_f, b_n, b_f, l_n, l_f)
        
        x = (t_n + t_f + r_n + r_f + b_n + b_f + l_n + l_f) / 8.0
        x = self.fc(x)
        return x
    
    
class OctaveNetEfficient(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Octave Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        size = 3
        self.initConv = ImageToOctaveEfficientConv(size, blockSize)
        size += blockSize
        
        for i in range(numBlocks):
            self.convList1.append(OctaveToOctaveEfficientConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(OctaveToOctaveEfficientConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(OctaveToOctaveEfficientConv(size, blockSize))
            size += blockSize
        
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        y = self.initConv(x)
        x = catReluSingleMultiple(x, y)
        print(x.shape)
        
        for i in range(self.numBlocks):
            y = self.convList1[i](x)
            x = catReluMultipleMultiple(x, y)
        print(x.shape)
        
        x = avgPool2d(x)
        
        for i in range(self.numBlocks):
            y = self.convList2[i](x)
            x = catReluMultipleMultiple(x, y)
        print(x.shape)
        
        x = avgPool2d(x)
        
        for i in range(self.numBlocks):
            y = self.convList3[i](x)
            x = catReluMultipleMultiple(x, y)
        print(x.shape)
        
        x = avgPool2d(x, kSize = 8)
        x = x.view(*x.shape[:2])
        print(x.shape)
        x = averageMultiple(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        return x
    
  
    
class MaxColumnPool(nn.Module):
    def __init__(self, inputDepth, poolSize):
        super().__init__()
        self.inputDepth = inputDepth
        #self.conv1d = nn.Conv2d(inputDepth, 1, 1, bias = False)
        self.poolSize = poolSize
        
    def forward(self, x):
        mb = x.shape[0]
        y = x.mean(1, keepdim=True)
        #print(y)
        _, max_ind = F.max_pool2d(y, self.poolSize, return_indices = True)
        #print(max_ind)
        z = y.new_zeros(y.shape)
        z.view(mb, -1)[torch.arange(mb).view(mb, 1), max_ind.view(mb, -1)] = 1
        #print(z)
        x = x * z * (self.poolSize ** 2)
        #print(x)
        return F.avg_pool2d(x, self.poolSize)

class MaxColumnNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools):
        super().__init__()
        self.netName = "Max Column Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            self.pools.append(MaxColumnPool(size, self.poolingSizes[layer]))
            #self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        x = self.fc(x.squeeze())
        return x
    

    
    
class QuadraticConv(nn.Module):
    def __init__(self, inputDepth, outputDepth, bottleneckDepth):
        super().__init__()
        self.bottleConv = nn.Conv2d(inputDepth, bottleneckDepth, (3,3), padding = 1)
        self.Wq = Parameter(torch.randn(outputDepth, bottleneckDepth, bottleneckDepth) * 0.5 / inputDepth)
        self.conv = nn.Conv2d(inputDepth, outputDepth, (3,3), padding = 1)
        
    def forward(self, x):
        convRes = self.conv(x)
        y = self.bottleConv(x)
        z = torch.einsum('abcd, ebf, afcd -> aecd', y, self.Wq, y)
        return z + convRes
    
class QuadraticNet(BaseNet):
    def __init__(self, numBlocks, blockSize, bottleneckDepth, numPools = 3):
        super().__init__()
        self.netName = "Quadratic Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(QuadraticConv(size, blockSize, bottleneckDepth))               
                size += blockSize
            self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
            #self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
        self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        x = self.fc(x.squeeze())
        return x
        
class PositionDependentPool(nn.Module):
    def __init__(self, rows = 8, cols = 8):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.weights = Parameter(torch.randn(rows * cols))
        
    def forward(self, x):
        mb, depth, rows, cols = x.shape
        w = F.softmax(self.weights).view(self.rows, self.cols)
        return (x * w).view(mb, depth, -1).sum(dim = -1)

class EquivariantNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3):
        super().__init__()
        self.netName = "Equivariant Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            if (layer != numPools - 1):
                self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
            else:
                #self.pools.append(PositionDependentPool(self.poolingSizes[layer], self.poolingSizes[layer]))
                self.pools.append(Identity())
            #self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
        self.fc = nn.Linear(size * (int(32 / (2**(numPools-1))))**2, 10)
        
    def forward(self, x):
        mb = x.shape[0]
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        x = x.view(mb, -1)
        x = self.fc(x)
        return x
    

            
            
class ProdPoolNet(BaseNet):
    def __init__(self, numBlocks, blockSize, numPools = 3):
        super().__init__()
        self.netName = "Usual Dense Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.numPools = numPools
        
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(numPools):
            self.convs.append(nn.ModuleList())
            
        self.poolingSizes = [2] * (numPools-1) + [int(32 / (2**(numPools-1)))]
        
        size = 3
        for layer in range(numPools):
            for block in range(numBlocks):
                self.convs[layer].append(nn.Conv2d(size, blockSize, 3, padding = 1))               
                size += blockSize
            if (layer != numPools-1):
                self.pools.append(nn.AvgPool2d(self.poolingSizes[layer]))
            else:
                self.pools.append(ProductPool(size))
        #self.fc = nn.Linear(size, 10)
        
    def forward(self, x):
        for layer in range(self.numPools):
            for block in range(self.numBlocks):
                x = torch.cat((x, F.relu(self.convs[layer][block](x))), dim = 1)
            x = self.pools[layer](x)
        #x = self.fc(x.squeeze())
        return x
    
    def freezeConvLayers(self):
        for param in self.convs.parameters():
            param.requires_grad = False
            
    def getLoss(self, reduction = "mean"):
        return nn.NLLLoss(reduction = reduction)
    
    
    
    
    
class PointToVectorConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        #self.depth = depth
        self.convParams = Parameter((torch.rand(outputDepth, inputDepth, 2)-0.5)*0.1)
        self.basis_x = torch.Tensor([[[1, 0, -1], [0, 0, 0], [1, 0, -1]], 
                                     [[0, 0, 0], [1, 0, -1], [0, 0, 0]]])
        self.basis_y = torch.Tensor([[[-1, 0, -1], [0, 0, 0], [1, 0, 1]], 
                                     [[0, -1, 0], [0, 0, 0], [0, 1, 0]]])
        self.basis_x = Variable(self.basis_x, requires_grad = False)
        self.basis_y = Variable(self.basis_y, requires_grad = False)
        if torch.cuda.is_available():
            self.basis_x = self.basis_x.cuda()
            self.basis_y = self.basis_y.cuda()
        
        
    def forward(self, x):
        kernel_x = torch.einsum("ijk,kmn->ijmn", (self.convParams, self.basis_x))
        kernel_y = torch.einsum("ijk,kmn->ijmn", (self.convParams, self.basis_y))
        grad_x = F.conv2d(x, kernel_x, padding = 1)
        grad_y = F.conv2d(x, kernel_y, padding = 1)
        return grad_x, grad_y
    
    
class PointToVectHandcrafted(nn.Module):
    def __init__(self, inputDepth):
        super().__init__()
        self.convParams = Parameter((torch.rand(1, inputDepth, 1, 1)-0.5)*0.1)
        self.grad_x_kernel = Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), requires_grad=False).view(1,1,3,3)
        self.grad_y_kernel = Variable(torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), requires_grad=False).view(1,1,3,3)
        
        if torch.cuda.is_available():
            self.grad_x_kernel = self.grad_x_kernel.cuda()
            self.grad_y_kernel = self.grad_y_kernel.cuda()
    
    def forward(self, x):
        x = F.conv2d(x, self.convParams)
        grad_x = F.conv2d(x, self.grad_x_kernel, padding=1)
        grad_y = F.conv2d(x, self.grad_y_kernel, padding=1)
        return grad_x, grad_y
    
    

    
    
class Sobel(nn.Module):
    def __init__(self, inputDepth):
        super().__init__()
        self.grad_x_kernel = Variable(torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])\
                                      .repeat(inputDepth, 1, 1, 1), requires_grad=False)
        self.grad_y_kernel = Variable(torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])\
                                      .repeat(inputDepth, 1, 1, 1), requires_grad=False)
        
        if torch.cuda.is_available():
            self.grad_x_kernel = self.grad_x_kernel.cuda()
            self.grad_y_kernel = self.grad_y_kernel.cuda()
    
    def forward(self, x):
        grad_x = F.conv2d(x, self.grad_x_kernel, padding=1, groups = x.shape[1])
        grad_y = F.conv2d(x, self.grad_y_kernel, padding=1, groups = x.shape[1])
        return grad_x, grad_y
    
    

    


class ZeroSumNet(BaseNet):
    def __init__(self, numBlocks, blockSize, initialDepth):
        super().__init__()
        self.netName = "Zero Sum Net Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.initBlock = ZeroSumConv(3, initialDepth)
        
        size = initialDepth
        for i in range(numBlocks):
            self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1, bias = False))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(nn.Conv2d(size, blockSize, 3, padding = 1, bias = False))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(nn.Conv2d(size, blockSize, 3, padding = 1, bias = False))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        x = self.initBlock(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    
    
    
    
    

class GradNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Grad Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.gradBlock = Sobel(3)
        
        size = 6
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
        grad_x, grad_y = self.gradBlock(x)
        x = torch.cat((grad_x, grad_y), dim = 1)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
    
class SymmetricConv(nn.Module):
    def __init__(self, inputDepth, outDepth):
        super().__init__()
        self.a = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.b = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.c = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        
        
    def forward(self, x):
        k1 = torch.stack((self.a, self.b, self.a), dim=2)
        k2 = torch.stack((self.b, self.c, self.b), dim=2)
        k = torch.stack((k1, k2, k1), dim=3)
        return F.conv2d(x, k, padding=1)
    
    
class ZeroSumConv(nn.Module):
    def __init__(self, inputDepth, outDepth):
        super().__init__()
        self.a = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.b = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.c = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        self.d = Parameter(torch.randn(outDepth, inputDepth)* 0.5 / inputDepth)
        
        
    def forward(self, x):
        k1 = torch.stack((self.a, self.b, -self.a - self.b), dim=2)
        k2 = torch.stack((self.c, self.d, -self.c - self.d), dim=2)
        k3 = torch.stack((-self.a - self.c, -self.b - self.d, -self.a - self.b - self.c  -self.d), dim=2)
        k = torch.stack((k1, k2, k3), dim=3)
        return F.conv2d(x, k, padding=1)
    
    
class SymmetricNet(BaseNet):
    def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.netName = "Symmetric Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
       # self.gradBlock = Sobel(3)
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(SymmetricConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList2.append(SymmetricConv(size, blockSize))
            size += blockSize
            
        for i in range(numBlocks):
            self.convList3.append(SymmetricConv(size, blockSize))
            size += blockSize
        
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        #grad_x, grad_y = self.gradBlock(x)
        #x = torch.cat((grad_x, grad_y), dim = 1)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
    
    
    
    
class DistanceConv(nn.Module):
    def __init__(self, inpDepth, N, blockSize):
        super().__init__()
        self.N = N
        self.blockSize = blockSize
        self.neighbors = Parameter((torch.rand(N, 1, 9 * blockSize)-0.5)*0.1)
        self.conv = nn.Conv2d(inpDepth, blockSize, 3, padding = 1)
        
    def forward(self, x):
        x = self.conv(x)
        mb, _, rows, cols = list(x.shape)
        #assert (depth == self.inpDepth)
        x = torch.abs(F.unfold(x, (3,3), padding=1).transpose(-1, -2).unsqueeze(1) - self.neighbors)\
        .mean(dim = -1).view(mb, self.N, rows, cols)
        return -x
    
def normalize(x):
    return x / torch.sqrt((x * x).mean(dim = -1, keepdim = True) + 0.001)

class DistanceVect(nn.Module):
    def __init__(self, inpLen, N, nClasses = 10):
        super().__init__()
        self.neighbors = Parameter((torch.rand(nClasses, N, inpLen)-0.5)*0.1)
        
    def forward(self, x):
        x = normalize(x)
        n_n = normalize(self.neighbors)
        x = x.unsqueeze(1).unsqueeze(1)
        x = (x * n_n).mean(dim = -1).max(dim = -1)[0]
        #x = ((x - self.neighbors)**2).mean(dim = -1).min(dim = -1)[0]
        return x
    
    
class DistanceVectNet(BaseNet):
    def __init__(self, numBlocks, blockSize, N):
        super().__init__()
        self.netName = "Distance Vect Net"
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
        if (N>0):
            self.dist = DistanceVect(size, N)
        else:
            self.dist = nn.Linear(size, 10)
        
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
        x = x.squeeze()
        x = self.dist(x)
        return x
   
    
    
    
class DistanceNet(BaseNet):
    def __init__(self, numBlocks, N, blockSize = 1):
        super().__init__()
        self.netName = "Distance Net"
        self.numBlocks = numBlocks
        self.blockSize = blockSize
        self.N = N
        
        self.convList1 = nn.ModuleList()
        #self.convList2 = nn.ModuleList()
        #self.convList3 = nn.ModuleList()
        
        #self.initBlock = ZeroSumConv(3, initialDepth)
        
        size = 3
        for i in range(numBlocks):
            self.convList1.append(DistanceConv(size, N, blockSize))
            #self.convList1.append(nn.Conv2d(size, blockSize, 3, padding = 1, bias = False))
            size += N
            
        #for i in range(numBlocks):
        #    self.convList2.append(nn.Conv2d(size, blockSize, 3, padding = 1, bias = False))
        #    size += blockSize
            
        #for i in range(numBlocks):
        #    self.convList3.append(nn.Conv2d(size, blockSize, 3, padding = 1, bias = False))
        #    size += blockSize
        
        self.pool1 = nn.MaxPool2d(32, 32)
        #self.pool2 = nn.AvgPool2d(2, 2)
        #self.pool3 = nn.AvgPool2d(8, 8)
        self.fc = nn.Linear(size, 10)

    def forward(self, x):
        #x = self.initBlock(x)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, self.convList1[i](x)), dim = 1)
        x = self.pool1(x)
        
        #for i in range(self.numBlocks):
        #    x = torch.cat((x, F.relu(self.convList2[i](x))), dim = 1)
        #x = self.pool2(x)
        #for i in range(self.numBlocks):
        #    x = torch.cat((x, F.relu(self.convList3[i](x))), dim = 1)
        #x = self.pool3(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

    
    
    
    
        
        
class RotationalConv(nn.Module):
    def __init__(self, inpDepth, outDepth):
        super().__init__()
        self.inpDepth = inpDepth
        self.outDepth = outDepth
        self.kernel = Parameter((torch.rand(outDepth, inpDepth, 3, 3)-0.5)*0.1)
        
    def forward(self, x, vx, vy):
        mb, depth, rows, cols = list(x.shape)
        assert (depth == self.inpDepth)
        vx = vx.view(mb, 1, rows, cols)
        vy = vy.view(mb, 1, rows, cols)
        
        l = torch.sqrt(vx * vx + vy * vy + 0.1)
        vx = vx / l
        vy = vy / l
        
        x_unfold = F.unfold(x, (5,5), padding=2).transpose(-1,-2).contiguous().view(mb*rows*cols, depth, 5, 5)
        
        theta1 = torch.stack((vx, vy, torch.zeros_like(vx)), dim=-1)
        theta2 = torch.stack((-vy, vx, torch.zeros_like(vx)), dim=-1)
        theta = torch.stack((theta1, theta2), dim=-2).view(-1, 2, 3)

        grid = F.affine_grid(theta, torch.Size((mb * rows * cols, depth, 3, 3)))
        result = F.grid_sample(x_unfold, grid)
        result_new = result.view(mb, rows, cols, depth, 3, 3)
        
        final_result = torch.einsum('abcdef,gdef->agbc', result_new, self.kernel)
        
        return final_result
        
class VectorToVectorConv(nn.Module):
    def __init__(self, inputDepth, outputDepth):
        super().__init__()
        #self.depth = depth
        self.convParams_1 = Parameter((torch.rand(outputDepth, inputDepth, 4)-0.5)*0.1)
        self.convParams_2 = Parameter((torch.rand(outputDepth, inputDepth, 1)-0.5)*0.1)

        self.basis_xx = torch.Tensor([[[1, 0, 1], [0, 0, 0], [1, 0, 1]], 
                                     [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
                                     [[0, 0, 0], [1, 0, 1], [0, 0, 0]], 
                                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        self.basis_yy = torch.Tensor([[[1, 0, 1], [0, 0, 0], [1, 0, 1]], 
                                     [[0, 0, 0], [1, 0, 1], [0, 0, 0]], 
                                     [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
                                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        self.basis_xy = torch.Tensor([[[1, 0, -1], [0, 0, 0], [-1, 0, 1]]])
        
        self.basis_xx = Variable(self.basis_xx, requires_grad = False)
        self.basis_yy = Variable(self.basis_yy, requires_grad = False)
        self.basis_xy = Variable(self.basis_xy, requires_grad = False)
        
        if torch.cuda.is_available():
            self.basis_xx = self.basis_xx.cuda()
            self.basis_xy = self.basis_xy.cuda()
            self.basis_yy = self.basis_yy.cuda()
        
        
    def forward(self, v_x, v_y):
        kernel_xx = torch.einsum("ijk,kmn->ijmn", (self.convParams_1, self.basis_xx))
        kernel_xy = torch.einsum("ijk,kmn->ijmn", (self.convParams_2, self.basis_xy))
        kernel_yy = torch.einsum("ijk,kmn->ijmn", (self.convParams_1, self.basis_yy))
        
        w_x = F.conv2d(v_x, kernel_xx, padding = 1) + F.conv2d(v_y, kernel_xy, padding = 1)
        w_y = F.conv2d(v_x, kernel_xy, padding = 1) + F.conv2d(v_y, kernel_yy, padding = 1)
        
        return w_x, w_y
    
    
    

    
    
    
    
class Nonlinearity(nn.Module):
    def __init__(self, inputDepth):
        super().__init__()
        self.params = Parameter(torch.rand(1, inputDepth, 1, 1)*0.1)
        
    def forward(self, v_x, v_y):
        #print(v_x.shape)
        d = torch.sqrt(v_x*v_x+v_y*v_y+0.001)
        d_new = F.relu(d - self.params) + self.params
        return v_x/d*d_new, v_y/d*d_new
    
class VectToPoint(nn.Module):
    def __init__(self, inputDepth):
        super().__init__()
        self.inputDepth = inputDepth
        self.kernel = torch.Tensor([[1,1,1],[1,0,1],[1,1,1]]).repeat(inputDepth,1,1).view(inputDepth, 1, 3, 3)
        self.kernel = Variable(self.kernel, requires_grad = False)
        
    def forward(self, v_x, v_y):
        v_x_avg = F.conv2d(v_x, self.kernel, groups=self.inputDepth, padding=1)
        v_y_avg = F.conv2d(v_y, self.kernel, groups=self.inputDepth, padding=1)
        return v_x*v_x_avg + v_y * v_y_avg
    
    
    
class InvariantNet(BaseNet):
     def __init__(self, numBlocks, blockSize):
        super().__init__()
        self.initPool = nn.AvgPool2d(8, 8)
        self.gradPool_x = nn.AvgPool2d(4, 4)
        self.gradPool_y = nn.AvgPool2d(4, 4)
        self.initGrad = PointToVectHandcrafted(3)
        self.pad = nn.ConstantPad2d(8, 0.)
        
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
        mb, depth, rows, cols = list(x.shape)
        
        pooled = self.initPool(x)
        grad_x, grad_y = self.initGrad(pooled)
        
        grad_x = self.gradPool_x(grad_x).squeeze()
        grad_y = self.gradPool_x(grad_y).squeeze()
        
        l = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 0.1)
        grad_x = grad_x / l
        grad_y = grad_y / l
        
        theta1 = torch.stack((grad_x, grad_y, torch.zeros_like(grad_x)), dim=-1)
        theta2 = torch.stack((-grad_y, grad_x, torch.zeros_like(grad_x)), dim=-1)
        theta = torch.stack((theta1, theta2), dim=-2).view(-1, 2, 3)
        
        grid = F.affine_grid(theta, torch.Size((mb, 3, rows, cols)))
        
        x = self.pad(x)
        
        x = F.grid_sample(x, grid)
        
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList1[i](x))), dim = 1)
        x = self.pool1(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList2[i](x))), dim = 1)
        x = self.pool2(x)
        for i in range(self.numBlocks):
            x = torch.cat((x, F.relu(self.convList3[i](x))), dim = 1)
        x = self.pool3(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
        
        
        
        
        
        
        
        
        
        
    

class RotationalNet(BaseNet):
    def __init__(self, nLayers, layerSize):
        super().__init__()
        self.netName = "Simplified Angle Net"
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.pointList1 = nn.ModuleList()
        self.pointList2 = nn.ModuleList()
        self.pointList3 = nn.ModuleList()
        
        self.vectList1 = nn.ModuleList()
        self.vectList2 = nn.ModuleList()
        self.vectList3 = nn.ModuleList()
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        pointDepth = 3
        self.initConv = PointToVectorConv(pointDepth, 1)
        vectDepth = 1
        
        for i in range(self.nLayers):
            self.pointList1.append(PointToVectorConv(pointDepth, 1))
            self.vectList1.append(VectorToVectorConv(vectDepth, 1))
            self.convList1.append(RotationalConv(pointDepth, layerSize))
            pointDepth += self.layerSize
            vectDepth += 1
            
            
        for i in range(self.nLayers):
            self.pointList2.append(PointToVectorConv(pointDepth, 1))
            self.vectList2.append(VectorToVectorConv(vectDepth, 1))
            self.convList2.append(RotationalConv(pointDepth, layerSize))
            pointDepth += self.layerSize
            vectDepth += 1
            
            
        for i in range(self.nLayers):
            self.pointList3.append(PointToVectorConv(pointDepth, 1))
            self.vectList3.append(VectorToVectorConv(vectDepth, 1))
            self.convList3.append(RotationalConv(pointDepth, layerSize))
            pointDepth += self.layerSize
            vectDepth += 1
            
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)        
        
        self.fc1 = nn.Linear(pointDepth, 10)
        self.fc2 = nn.Linear(vectDepth, 10)
        self.fc3 = nn.Linear(vectDepth, 10)
        
    def forward(self, x):
        v_x, v_y = self.initConv(x)
        for i in range(self.nLayers):
            w_x, w_y = self.pointList1[i](x)
            t_x, t_y = self.vectList1[i](v_x, v_y)
            w_x += t_x
            w_y += t_y
            h = F.relu(self.convList1[i](x, w_x, w_y))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool1(x)
        v_x = self.pool1(v_x)
        v_y = self.pool1(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList2[i](x)
            t_x, t_y = self.vectList2[i](v_x, v_y)
            w_x += t_x
            w_y += t_y
            h = F.relu(self.convList2[i](x, w_x, w_y))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool2(x)
        v_x = self.pool2(v_x)
        v_y = self.pool2(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList3[i](x)
            t_x, t_y = self.vectList3[i](v_x, v_y)
            w_x += t_x
            w_y += t_y
            h = F.relu(self.convList3[i](x, w_x, w_y))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool3(x).squeeze()
        v_x = self.pool3(v_x).squeeze()
        v_y = self.pool3(v_y).squeeze()
        
        x = self.fc1(x)+self.fc2(v_x)+self.fc3(v_y)
        return x
    
    
class DerotateVectors(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, vx, vy):
        vx_avg = torch.mean(vx, dim = 1, keepdim = True)
        vy_avg = torch.mean(vy, dim = 1, keepdim = True)
        
        l = torch.sqrt(vx_avg * vx_avg + vy_avg * vy_avg + 0.01)
        
        vx_avg = vx_avg / l
        vy_avg = vy_avg / l
        
        vx_res =  vx * vx_avg + vy * vy_avg
        vy_res = -vx * vy_avg + vy * vx_avg
        
        return vx_res, vy_res
    
    

    
    
class SimplifiedRotationalNet(BaseNet):
    def __init__(self, nLayers, layerSize):
        super().__init__()
        self.netName = "Simplified Angle Net"
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.pointList1 = nn.ModuleList()
        self.pointList2 = nn.ModuleList()
        self.pointList3 = nn.ModuleList()
        
        self.conv1dList1 = nn.ModuleList()
        self.conv1dList2 = nn.ModuleList()
        self.conv1dList3 = nn.ModuleList()
        
        #self.vectList1 = nn.ModuleList()
        #self.vectList2 = nn.ModuleList()
        #self.vectList3 = nn.ModuleList()
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.BNList1 = nn.ModuleList()
        self.BNList2 = nn.ModuleList()
        self.BNList3 = nn.ModuleList()
        
        self.derotate = DerotateVectors()
        
        
        pointDepth = 3
        self.initConv = PointToVectorConv(pointDepth, 1)
        vectDepth = 1
        
        for i in range(self.nLayers):
            self.pointList1.append(PointToVectHandcrafted(pointDepth))
            self.conv1dList1.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList1.append(VectorToVectorConv(vectDepth, 1))
            self.convList1.append(RotationalConv(layerSize, layerSize))
            self.BNList1.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += 1
            
            
        for i in range(self.nLayers):
            self.pointList2.append(PointToVectHandcrafted(pointDepth))
            self.conv1dList2.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList2.append(VectorToVectorConv(vectDepth, 1))
            self.convList2.append(RotationalConv(layerSize, layerSize))
            self.BNList2.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += 1
            
            
        for i in range(self.nLayers):
            self.pointList3.append(PointToVectHandcrafted(pointDepth))
            self.conv1dList3.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList3.append(VectorToVectorConv(vectDepth, 1))
            self.convList3.append(RotationalConv(layerSize, layerSize))
            self.BNList3.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += 1
            
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)        
        
        self.fc1 = nn.Linear(pointDepth, 10)
        self.fc2 = nn.Linear(vectDepth, 10)
        self.fc3 = nn.Linear(vectDepth, 10)
        
    def forward(self, x):
        v_x, v_y = self.initConv(x)
        for i in range(self.nLayers):
            w_x, w_y = self.pointList1[i](x)
            #t_x, t_y = self.vectList1[i](v_x, v_y)
            #w_x += t_x
            #w_y += t_y
            z = self.conv1dList1[i](x)
            h = F.relu(self.BNList1[i](self.convList1[i](z, w_x, w_y)))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool1(x)
        v_x = self.pool1(v_x)
        v_y = self.pool1(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList2[i](x)
            #t_x, t_y = self.vectList2[i](v_x, v_y)
            #w_x += t_x
            #w_y += t_y
            z = self.conv1dList2[i](x)
            h = F.relu(self.BNList2[i](self.convList2[i](z, w_x, w_y)))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool2(x)
        v_x = self.pool2(v_x)
        v_y = self.pool2(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList3[i](x)
            #t_x, t_y = self.vectList3[i](v_x, v_y)
            #w_x += t_x
            #w_y += t_y
            z = self.conv1dList3[i](x)
            h = F.relu(self.BNList3[i](self.convList3[i](z, w_x, w_y)))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool3(x).squeeze()
        v_x = self.pool3(v_x).squeeze()
        v_y = self.pool3(v_y).squeeze()
        
        v_x, v_y = self.derotate(v_x, v_y)
        
        x = self.fc1(x)+self.fc2(v_x)+self.fc3(v_y)
        return x
        
        
        

        
class SymmetricRotationalNet(BaseNet):
    def __init__(self, nLayers, layerSize):
        super().__init__()
        self.netName = "Simplified Angle Net"
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.pointList1 = nn.ModuleList()
        self.pointList2 = nn.ModuleList()
        self.pointList3 = nn.ModuleList()
        
        self.conv1dList1 = nn.ModuleList()
        self.conv1dList2 = nn.ModuleList()
        self.conv1dList3 = nn.ModuleList()
        
        #self.vectList1 = nn.ModuleList()
        #self.vectList2 = nn.ModuleList()
        #self.vectList3 = nn.ModuleList()
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.BNList1 = nn.ModuleList()
        self.BNList2 = nn.ModuleList()
        self.BNList3 = nn.ModuleList()
        
        self.derotate = DerotateVectors()
        
        
        pointDepth = 3
        self.initConv = Sobel(pointDepth)
        vectDepth = 3
        
        for i in range(self.nLayers):
            self.pointList1.append(Sobel(layerSize))
            self.conv1dList1.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList1.append(VectorToVectorConv(vectDepth, 1))
            self.convList1.append(SymmetricConv(layerSize, layerSize))
            self.BNList1.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += layerSize
            
            
        for i in range(self.nLayers):
            self.pointList2.append(Sobel(layerSize))
            self.conv1dList2.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList2.append(VectorToVectorConv(vectDepth, 1))
            self.convList2.append(SymmetricConv(layerSize, layerSize))
            self.BNList2.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += layerSize
            
            
        for i in range(self.nLayers):
            self.pointList3.append(Sobel(layerSize))
            self.conv1dList3.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList3.append(VectorToVectorConv(vectDepth, 1))
            self.convList3.append(SymmetricConv(layerSize, layerSize))
            self.BNList3.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += layerSize
            
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)        
        
        self.fc1 = nn.Linear(pointDepth, 10)
        self.fc2 = nn.Linear(vectDepth, 10)
        self.fc3 = nn.Linear(vectDepth, 10)
        
    def forward(self, x):
        v_x, v_y = self.initConv(x)
        for i in range(self.nLayers):
            z = self.conv1dList1[i](x)
            h = F.relu(self.BNList1[i](self.convList1[i](z)))
            w_x, w_y = self.pointList1[i](h)
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool1(x)
        v_x = self.pool1(v_x)
        v_y = self.pool1(v_y)
        
        for i in range(self.nLayers):
            z = self.conv1dList2[i](x)
            h = F.relu(self.BNList2[i](self.convList2[i](z)))
            w_x, w_y = self.pointList2[i](h)
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool2(x)
        v_x = self.pool2(v_x)
        v_y = self.pool2(v_y)
        
        for i in range(self.nLayers):
            z = self.conv1dList3[i](x)
            h = F.relu(self.BNList3[i](self.convList3[i](z)))
            w_x, w_y = self.pointList3[i](h)
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool3(x).squeeze()
        v_x = self.pool3(v_x).squeeze()
        v_y = self.pool3(v_y).squeeze()
        
        v_x, v_y = self.derotate(v_x, v_y)
        
        x = self.fc1(x)+self.fc2(v_x)+self.fc3(v_y)
        return x
        
        
class SimplifiedRotationalNet(BaseNet):
    def __init__(self, nLayers, layerSize):
        super().__init__()
        self.netName = "Simplified Angle Net"
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.pointList1 = nn.ModuleList()
        self.pointList2 = nn.ModuleList()
        self.pointList3 = nn.ModuleList()
        
        self.conv1dList1 = nn.ModuleList()
        self.conv1dList2 = nn.ModuleList()
        self.conv1dList3 = nn.ModuleList()
        
        #self.vectList1 = nn.ModuleList()
        #self.vectList2 = nn.ModuleList()
        #self.vectList3 = nn.ModuleList()
        
        self.convList1 = nn.ModuleList()
        self.convList2 = nn.ModuleList()
        self.convList3 = nn.ModuleList()
        
        self.BNList1 = nn.ModuleList()
        self.BNList2 = nn.ModuleList()
        self.BNList3 = nn.ModuleList()
        
        self.derotate = DerotateVectors()
        
        
        pointDepth = 3
        self.initConv = PointToVectorConv(pointDepth, 1)
        vectDepth = 1
        
        for i in range(self.nLayers):
            self.pointList1.append(PointToVectHandcrafted(pointDepth))
            self.conv1dList1.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList1.append(VectorToVectorConv(vectDepth, 1))
            self.convList1.append(RotationalConv(layerSize, layerSize))
            self.BNList1.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += 1
            
            
        for i in range(self.nLayers):
            self.pointList2.append(PointToVectHandcrafted(pointDepth))
            self.conv1dList2.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList2.append(VectorToVectorConv(vectDepth, 1))
            self.convList2.append(RotationalConv(layerSize, layerSize))
            self.BNList2.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += 1
            
            
        for i in range(self.nLayers):
            self.pointList3.append(PointToVectHandcrafted(pointDepth))
            self.conv1dList3.append(nn.Conv2d(pointDepth, layerSize, 1))
            #self.vectList3.append(VectorToVectorConv(vectDepth, 1))
            self.convList3.append(RotationalConv(layerSize, layerSize))
            self.BNList3.append(nn.BatchNorm2d(num_features = layerSize))
            pointDepth += layerSize
            vectDepth += 1
            
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)        
        
        self.fc1 = nn.Linear(pointDepth, 10)
        self.fc2 = nn.Linear(vectDepth, 10)
        self.fc3 = nn.Linear(vectDepth, 10)
        
    def forward(self, x):
        v_x, v_y = self.initConv(x)
        for i in range(self.nLayers):
            w_x, w_y = self.pointList1[i](x)
            #t_x, t_y = self.vectList1[i](v_x, v_y)
            #w_x += t_x
            #w_y += t_y
            z = self.conv1dList1[i](x)
            h = F.relu(self.BNList1[i](self.convList1[i](z, w_x, w_y)))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool1(x)
        v_x = self.pool1(v_x)
        v_y = self.pool1(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList2[i](x)
            #t_x, t_y = self.vectList2[i](v_x, v_y)
            #w_x += t_x
            #w_y += t_y
            z = self.conv1dList2[i](x)
            h = F.relu(self.BNList2[i](self.convList2[i](z, w_x, w_y)))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool2(x)
        v_x = self.pool2(v_x)
        v_y = self.pool2(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList3[i](x)
            #t_x, t_y = self.vectList3[i](v_x, v_y)
            #w_x += t_x
            #w_y += t_y
            z = self.conv1dList3[i](x)
            h = F.relu(self.BNList3[i](self.convList3[i](z, w_x, w_y)))
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool3(x).squeeze()
        v_x = self.pool3(v_x).squeeze()
        v_y = self.pool3(v_y).squeeze()
        
        v_x, v_y = self.derotate(v_x, v_y)
        
        x = self.fc1(x)+self.fc2(v_x)+self.fc3(v_y)
        return x
        
        
        
        
        
        
        
class SimplifiedAngleNet(BaseNet):
    def __init__(self, nLayers, layerSize):
        super().__init__()
        self.netName = "Simplified Angle Net"
        self.nLayers = nLayers
        self.layerSize = layerSize
        
        self.pointList1 = nn.ModuleList()
        self.pointList2 = nn.ModuleList()
        self.pointList3 = nn.ModuleList()
        
        self.vectList1 = nn.ModuleList()
        self.vectList2 = nn.ModuleList()
        self.vectList3 = nn.ModuleList()
        
        self.normList1 = nn.ModuleList()
        self.normList2 = nn.ModuleList()
        self.normList3 = nn.ModuleList()
        
        depth = 3
        self.initConv = PointToVectorConv(depth, depth)
        for i in range(self.nLayers):
            self.pointList1.append(PointToVectorConv(depth, self.layerSize))
            self.vectList1.append(VectorToVectorConv(depth, self.layerSize))
            self.normList1.append(VectToPoint(self.layerSize))
            depth += self.layerSize
            
            
        for i in range(self.nLayers):
            self.pointList2.append(PointToVectorConv(depth, self.layerSize))
            self.vectList2.append(VectorToVectorConv(depth, self.layerSize))
            self.normList2.append(VectToPoint(self.layerSize))
            depth += self.layerSize
            
            
        for i in range(self.nLayers):
            self.pointList3.append(PointToVectorConv(depth, self.layerSize))
            self.vectList3.append(VectorToVectorConv(depth, self.layerSize))
            self.normList3.append(VectToPoint(self.layerSize))
            depth += self.layerSize
            
            
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.pool3 = nn.AvgPool2d(8, 8)        
        
        self.fc1 = nn.Linear(depth, 10)
        self.fc2 = nn.Linear(depth, 10)
        self.fc3 = nn.Linear(depth, 10)
        
    def forward(self, x):
        v_x, v_y = self.initConv(x)
        for i in range(self.nLayers):
            w_x, w_y = self.pointList1[i](x)
            t_x, t_y = self.vectList1[i](v_x, v_y)
            w_x += t_x
            w_y += t_y
            h = self.normList1[i](w_x, w_y)
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool1(x)
        v_x = self.pool1(v_x)
        v_y = self.pool1(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList2[i](x)
            t_x, t_y = self.vectList2[i](v_x, v_y)
            w_x += t_x
            w_y += t_y
            h = self.normList2[i](w_x, w_y)
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool2(x)
        v_x = self.pool2(v_x)
        v_y = self.pool2(v_y)
        
        for i in range(self.nLayers):
            w_x, w_y = self.pointList3[i](x)
            t_x, t_y = self.vectList3[i](v_x, v_y)
            w_x += t_x
            w_y += t_y
            h = self.normList3[i](w_x, w_y)
            x = torch.cat((x, h), dim = 1)
            v_x = torch.cat((v_x, w_x), dim = 1)
            v_y = torch.cat((v_y, w_y), dim = 1)
        x = self.pool3(x).squeeze()
        v_x = self.pool3(v_x).squeeze()
        v_y = self.pool3(v_y).squeeze()
        
        
        
        x = self.fc1(x)+self.fc2(v_x)+self.fc3(v_y)
        return x
        
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
    def __init__(self, pDrop):
        super().__init__()
        self.pDrop = pDrop
        
    def forward(self, x):
        if (not self.training):
            return x
        if (self.pDrop < 0.001):
            return x
        #if (torch.bernoulli(torch.tensor([self.pNotDrop])).item()):
        #    return x
        active = x.new(x.shape[0], 1, x.shape[2], x.shape[3])
        active.bernoulli_(1.0 - self.pDrop).mul_(1.0/(1.0 - self.pDrop))
        return x.mul(active) 
    
class ProductPool(nn.Module):
    def __init__(self, inputDepth, outputDepth = 10):
        super().__init__()
        self.fc = nn.Linear(inputDepth, outputDepth)
        self.outputDepth = outputDepth
        
    def forward(self, x):
        #if (self.training):
            mb, depth, rows, cols = x.shape
            x = x.view(mb, depth, -1)
            x = x.transpose(-1, -2)
            x = x.contiguous().view(-1, depth)
            x = self.fc(x)
            x = torch.log(x.softmax(dim = -1) + 0.001)
            x = x.view(mb, -1, self.outputDepth)
            x = x.mean(dim = 1)
            return x
        #else:
        #    x = F.adaptive_avg_pool2d(x, 1).squeeze()
        #    x = self.fc(x)
        #    x = torch.log(x.softmax(dim = -1))
        #    return x
    
class OctaveColumnDrop(nn.Module):
    def __init__(self, pDrop, num = 8):
        super().__init__()
        self.pDrop = pDrop
        self.num = num
        
    def forward(self, x, num = 8):
        if (not self.training):
            return x
        if (self.pDrop < 0.001):
            return x
        mb, num_depth, rows, cols = x.shape
        x = x.view(mb, self.num, int(num_depth / self.num), rows, cols)
        active = x.new(mb, self.num, 1, rows, cols)
        active.bernoulli_(1.0 - self.pDrop).mul_(1.0/(1.0 - self.pDrop))
        x = x.mul(active)
        return x.view(mb, num_depth, rows, cols)
    
class OctaveDrop(nn.Module):
    def __init__(self, pDrop, num = 8):
        super().__init__()
        self.num = num
        self.pDrop = pDrop
        
    def forward(self, x):
        if (not self.training):
            return x
        if (self.pDrop < 0.001):
            return x
        mb, num_depth = x.shape[0], x.shape[1]
        shape = x.shape
        x = x.view(mb, self.num, int(num_depth / self.num), *shape[2:])
        active = x.new(mb, self.num, 1, *((1,)*len(shape[2:])))
        active.bernoulli_(1.0 - self.pDrop).mul_(1.0/(1.0 - self.pDrop))
        x = x.mul(active)
        return x.view(*shape)
    
    

    
    
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
    
