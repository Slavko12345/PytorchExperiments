import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR 
import time
import DenseModule
import sys
import numpy as np

def defineDataLoaders(dataset, trainBatchSize, testBatchSize, dataAugmentation):
    loaders = {"Cifar-10": getCifar10DataLoaders,
               "MnistRot": getMnistRotDataLoaders,
               "Cifar-100": getCifar100DataLoaders}
    return loaders[dataset](trainBatchSize, testBatchSize, dataAugmentation)

def getMnistRotDataLoaders(trainBatchSize, testBatchSize, dataAugmentation):
    mnist_dir = 'data/mnist_rot'
    train = np.load(mnist_dir + '/rotated_train.npz')
    valid = np.load(mnist_dir + '/rotated_valid.npz')
    test = np.load(mnist_dir + '/rotated_test.npz')
    data = {}
    data['train_x'] = torch.tensor(np.vstack((train['x'], valid['x'])).reshape(-1, 1, 28, 28))
    data['train_y'] = torch.tensor(np.hstack((train['y'], valid['y']))).long()
    data['test_x']  = torch.tensor(test['x'].reshape(-1, 1, 28, 28))
    data['test_y']  = torch.tensor(test['y']).long()
    #if (torch.cuda.is_available()):
    #    for key in data.keys():
    #        data[key] = data[key].cuda()
    trainset = torch.utils.data.TensorDataset(data['train_x'], data['train_y'])
    testset =  torch.utils.data.TensorDataset(data['test_x'],  data['test_y'])
    train_loader      = torch.utils.data.DataLoader(trainset, batch_size = trainBatchSize, shuffle = True,  num_workers = 4)
    train_test_loader = torch.utils.data.DataLoader(trainset, batch_size = trainBatchSize, shuffle = False, num_workers = 4)
    test_loader       = torch.utils.data.DataLoader(testset , batch_size = testBatchSize,  shuffle = False, num_workers = 4)
    classes = tuple([str(i) for i in range(10)])
    shape = (1,28,28)
    return train_loader, train_test_loader, test_loader, classes, shape

#import, load and normalize CIFAR
def getCifar10DataLoaders(trainBatchSize, testBatchSize, dataAugmentation):
    cuda_aval = torch.cuda.is_available()
    
    transform_train = None
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if (dataAugmentation):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transform_test
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize,
                                              shuffle=True, num_workers = 4, pin_memory = True)

    train_test_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_test)
    train_test_loader = torch.utils.data.DataLoader(train_test_set, batch_size=trainBatchSize,
                                              shuffle=False,  num_workers = 4, pin_memory = True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize,
                                             shuffle=False,  num_workers = 4, pin_memory = True)

    classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    shape = (3, 32, 32)
    return trainloader, train_test_loader, testloader, classes, shape


def getCifar100DataLoaders(trainBatchSize, testBatchSize, dataAugmentation):
    cuda_aval = torch.cuda.is_available()
    
    transform_train = None
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    if (dataAugmentation):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform_train = transform_test
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize,
                                              shuffle=True, num_workers = 4, pin_memory = True)

    train_test_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_test)
    train_test_loader = torch.utils.data.DataLoader(train_test_set, batch_size=trainBatchSize,
                                              shuffle=False,  num_workers = 4, pin_memory = True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=testBatchSize,
                                             shuffle=False,  num_workers = 4, pin_memory = True)

    classes = tuple([str(i) for i in range(100)])
    shape = (3, 32, 32)
    return trainloader, train_test_loader, testloader, classes, shape


def testPerformance(net, dataLoader, dataName):
    correct = 0
    total = 0
    loss = 0
    testCriterion = net.getLoss(reduction = 'sum')
    cuda_available = torch.cuda.is_available()
    
    with torch.no_grad():
        net.eval()
        for data in dataLoader:
            inputs, labels = data
            if cuda_available:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            labels_converted = net.convertLabels(labels)
            outputs = net(inputs)
            loss += testCriterion(outputs, labels_converted)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss = loss / total
    acc = 100 * correct / total
    print('Loss and accuracy on the %s set: %.5f, %.2f %%' % (dataName, loss, acc))
    return loss, acc

def adjust_learning_rate(optimizer, new_learning_rate):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_learning_rate
            
def trainNetwork(net, learning_rate, lr_decay, numEpochs, ALPHA_CHANGING, MIN_ALPHA, MAX_ALPHA, trainloader, train_test_loader, testloader):
    #Train on training set
    criterion = net.getLoss()

    #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 5e-4)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    #scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        net.cuda()

    net.printNet()
    print("Number of epochs: ", numEpochs)

    flag = 0
    alpha = 10**(MIN_ALPHA)
    
    trainTimeStart = time.time()
    for epoch in range(numEpochs):  # loop over the dataset multiple times
        timeStart = time.time()
        #scheduler.step()
        net.train()
        running_loss = 0.0
        numBatches = 0
        if (ALPHA_CHANGING):
            alpha = 10 ** ((epoch / numEpochs) * (MAX_ALPHA - MIN_ALPHA) + MIN_ALPHA)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            if cuda_avail:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            
            labels_converted = net.convertLabels(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if (ALPHA_CHANGING):
                outputs = net(inputs, alpha)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, labels_converted)

            '''if (flag == 0):
                save('s.dot', loss.grad_fn)
                flag = 1'''

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            numBatches += 1
        timeElapsed = time.time() - timeStart
        
        learning_rate *= lr_decay
        adjust_learning_rate(optimizer, learning_rate)

        print('[%d] loss: %.3f LR: %.5f Epoch time: %.2f s, Remaining time: %.2f s alpha: %.2f' %
              (epoch + 1, running_loss / numBatches, learning_rate, timeElapsed, (numEpochs - epoch - 1) * timeElapsed, alpha))

        if ((epoch + 1) % 10 == 0):
            testPerformance(net, train_test_loader, "train")
            testPerformance(net, testloader, "test")
    trainDuration = time.time() - trainTimeStart
    print('Finished Training')
    train_loss, train_acc = testPerformance(net, train_test_loader, "train")
    test_loss, test_acc = testPerformance(net, testloader, "test")
    return train_loss, train_acc, test_loss, test_acc, trainDuration / numEpochs


def calculateDistributionOverClasses(net, testloader, classes):
    #Distribution over classes in test set
    if (len(classes)>10):
        return
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    numClasses = len(classes)
    #class_correct = list(0. for i in range(10))
    #class_total = list(0. for i in range(10))
    cuda_available = torch.cuda.is_available()
    classDistr = torch.zeros(numClasses, numClasses)
    classTotal = torch.zeros(numClasses)
    with torch.no_grad():
        net.eval()
        for data in testloader:
            inputs, labels = data
            if cuda_available:        
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            #c = (predicted == labels).squeeze()
            for i in range(predicted.shape[0]):
                label = labels[i]
                pred = predicted[i]
                classDistr[label][pred] += 1
                classTotal[label] += 1
                #class_correct[label] += c[i].item()
                #class_total[label] += 1
    classDistr = 100 * classDistr / (classTotal.view(-1, 1))
    formatStr = "{:.2f}%\t" * numClasses
    
    print(("{}\t"*(1 + numClasses)).format(" ",*classes))
    for i, label in enumerate(classes):
        rowStr = label + ":\t" + formatStr.format(*classDistr[i].tolist())
        print(rowStr)
    
    
    
def testForSymmetryRotationInvariance(net, shape, numClasses):
    eps = 0.000001
    with torch.no_grad():
        x = torch.randn(1, *shape)
        if (torch.cuda.is_available()):
            x = x.cuda()
        net.eval()
        #np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        l1 = list(DenseModule.rotate4(x))
        l2 = list(DenseModule.flip_(*l1))
        l = l1 + l2
        outs = [net(inp) for inp in l]
        if (numClasses <= 10):
            for i in range(8):
                print(np.array(outs[i].tolist()))
         
        flipInv = (torch.abs(outs[0] - outs[4]) + torch.abs(outs[1] - outs[5]) + \
                  torch.abs(outs[2] - outs[6]) + torch.abs(outs[3] - outs[7]))
        
        rotInv = (torch.abs(outs[0] - outs[1]) + torch.abs(outs[1] - outs[2]) + \
                  torch.abs(outs[2] - outs[3]) + torch.abs(outs[4] - outs[5]) + \
                  torch.abs(outs[5] - outs[6]) + torch.abs(outs[6] - outs[7]))
        
        flipInv = bool((flipInv.mean() < eps).item())
        rotInv  = bool((rotInv.mean() < eps).item())
        return flipInv, rotInv
    
    
def testForAdditionInvariance(net):
    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32)
        randCh = torch.randn(1, 3, 1, 1)
        randHor = torch.randn(1, 3, 1, 32)
        randVert = torch.randn(1, 3, 32, 1)
        randDiag = torch.randn(3)
        if (torch.cuda.is_available()):
            x = x.cuda()
            randCh = randCh.cuda()
            randHor = randHor.cuda()
            randVert = randVert.cuda()
            randDiag = randDiag.cuda()

        net.eval()
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        xCh = x + randCh
        xHor = x + randHor
        xVert = x + randVert
        xDiag = x.clone()
        xDiag[0, torch.arange(3).view(3,1), torch.arange(32), torch.arange(32)] += randDiag.view(3,1)

        l = [x, xCh, xHor, xVert, xDiag]
        outs = [net(inp) for inp in l]
        modes = ["Plain", "Channel", "Horizontal", "Vertical", "Diagonal"]
        for out, mode in zip(outs, modes):
                print(mode, ": ", np.array(out.tolist()))
    
    
def trainNetworks(dataset, trainBatchSize, testBatchSize, dataAugmentation, netNames, learning_rate, lr_decay, numEpochs, ALPHA_CHANGING, MIN_ALPHA, MAX_ALPHA, testRotation = True, testAddition = False):
    trainloader, train_test_loader, testloader, classes, shape = defineDataLoaders(dataset, trainBatchSize, testBatchSize, dataAugmentation)
    results = []
    param_num = []
    for netName in netNames:
        netName = netName[:-1] + ", initDepth = " + str(shape[0]) + ", numClasses = " + str(len(classes)) + ")"
        command = "global net; net = DenseModule." + netName
        exec(command)
        net.netName = netName
        param_num.append(net.numTrainableParams())
        res = trainNetwork(net, learning_rate, lr_decay, numEpochs, ALPHA_CHANGING, MIN_ALPHA, MAX_ALPHA, trainloader, train_test_loader, testloader)
        calculateDistributionOverClasses(net, testloader, classes)
        if (testRotation):
            inv = testForSymmetryRotationInvariance(net, shape, len(classes))
        results.append(res + inv)
        if (testAddition):
            testForAdditionInvariance(net)
        print(netName + ": training is finished!")
    
    print(" \n Overall results: (train_loss, train_acc, test_loss, test_acc)")
    print("Dataset: ", dataset)
    print("Num of epochs: ", numEpochs)
    print("Learning rate %.4f, l.r. decay: %.3f" %(learning_rate, lr_decay))
    print("Minibatch size: ", trainBatchSize)
    print("\n")
    for netName, result, params in zip(netNames, results, param_num):
        #print(netName + ": " + result)
        print("%s: %d params" % (netName, params))
        print("%.5f, %.2f%%, %.5f, %.2f%%" % (result[0], result[1], result[2], result[3]))
        print("Average epoch duration: %.2f s" % result[4])
        print("Flip invariance: ", result[5])
        print("Rotation invariance: ", result[6])
        print("\n")