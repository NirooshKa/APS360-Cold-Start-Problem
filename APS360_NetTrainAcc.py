'''
This module contains possible nueral net archetechtures for the project, and the training and accuracy functions.
The modules include:
- FCTestNN
- ConvTestNN
- Net1
- Net2
(The networks increase in size and complexity as thier Net# increases)

The function names are:
- train_net
- accuracy_net
'''

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class FCTestNN(nn.Module):
    def __init__(self):
        super(FCTestNN, self).__init__()
        self.name = "FCTestNN"
        self.fc1 = nn.Linear(3*128*128, 256) #(RGB incoming pixels, arbitrary node number)
        self.fc2 = nn.Linear(256, 4) #(arbitrary node number, 4 classes (dog, cat, horse, flower))
    
    def forward(self, x):
        x = x.view(-1, 3*128*128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1)
        return x

class ConvTestNN(nn.Module):
    def __init__(self):
        super(ConvTestNN, self).__init__()
        self.name = "ConvTestNN"
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*62*62, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 5*62*62)
        x = self.fc1(x)
        x = x.squeeze(1)
        return x

class ALFinalClassifier(nn.Module):
    def __init__(self, class_size, pretrain_net):
        super(ALFinalClassifier, self).__init__()
        self.name = "ALFinalClassifier"
        self.fc1 = nn.Linear(256 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 64)
        self.fc3 = nn.Linear(64, class_size)
        self.pretrain_net = pretrain_net
    
    def forward(self, x):
        x = self.pretrain_net(x)
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x)) #Not sure if ReLu is the best function to use here, but i'll leave it for now
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x


def train_net(model, train_data, val_data, batch_size=4, num_epochs=1, learning_rate=0.001, momentum=0.9, use_cuda=False, num_iters=10, pretrain_net=None):
    #Initialize variables and loss/optim
    start_time = time.time()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    iters, losses, train_acc, val_acc = [], [], [], []

    n = 0 #iterations

    #Main loop
    for epoch in range(num_epochs):
        for imgs, labels in iter(train_data):

            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            #Forward pass
            out = model(imgs) #img size should be 224 * 224 for AlexNet
            loss = criterion(out, labels)

            #Backwards pass
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            if n % num_iters == 0:
                #Train and val loss
                print(num_iters)
                iters.append(n)
                losses.append(float(loss)/batch_size)

                #Train and val acc
                train_acc.append(accuracy_net(model, batch=imgs, label=labels))
                val_acc.append(accuracy_net(model, data=val_data, use_cuda=use_cuda))
            
            n += 1
        
        print("Epoch Number: {0}".format(epoch + 1))
    
    end_time = time.time()

    #Graph the training data
    plt.title("Training Loss")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training and Validation Accuracy")
    plt.plot(iters, train_acc, label="Train")
    plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.show()

    print("Final Training Accuracy: {0}".format(train_acc[-1]))
    print("Final Validation Accuracy: {0}".format(val_acc[-1]))
    print("The total training time was: {0}".format(end_time - start_time))

def accuracy_net(model, data=None, batch=None, label=None, use_cuda=False, pretrain_net=None):
    correct = 0
    total = 0

    #Check if input is DataLoader or an individual batch + labels
    if data is not None:
        for imgs, labels in data:
            #To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()

            #Model evaluation
            output = model(imgs)
            pred = output.max(1, keepdim=False)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += imgs.shape[0]

    else:
        #Model evaluation
        output = model(batch)
        pred = output.max(1, keepdim=False)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()
        total += batch.shape[0]

    return correct / total

def alexnet_init(use_cuda=False):
    alexNet = torchvision.models.alexnet(pretrained=True)
    Net = alexNet.features

    if use_cuda and torch.cuda.is_available():
        Net.cuda()
    
    return Net

