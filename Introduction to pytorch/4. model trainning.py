import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#get trainning & test dataset
transform=transforms.Compose(#make transform function
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)#make trainning set&loader
trainloader=torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)#make test set&loader
testloader=torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

classes=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#test output
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

dataiter=iter(trainloader)
images, labels=dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))

#model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3, 6, 5)#input3, output6, size5
        self.pool=nn.MaxPool2d(2, 2)#pooling over(2,2)
        self.conv2=nn.Conv2d(6,16, 5)#input6, output16, size5
        self.fc1=nn.Linear(16*5*5, 120)#output_of_Conv2d16 & dimention5*5, output120
        self.fc2=nn.Linear(120, 84)#input120, output84
        self.fc3=nn.Linear(84,10)#input84, output10_10 bins

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))#conv1 with pooling
        x=self.pool(F.relu(self.conv2(x)))#conv2...
        x=x.view(-1, 16*5*5)#reshaping to (auto, 16*5*5)
        x=F.relu(self.fc1(x))#use inner linear functions
        x=F.relu(self.fc2(x))
        x=self.fc3(x)#conv->pooling->FCL
        return x
    
net=Net()

#loss function & optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#momentum for fast learnning than other optimizer models

#trainning loop
for epoch in range(2):
    running_loss=0.0
    for i, data in enumerate(trainloader, 0):#index, data
        inputs, labels=data#predict value, correct value

        optimizer.zero_grad()

        outputs=net(inputs)
        loss=criterion(outputs, labels)#make lossfunction by criterion
        loss.backward()#calculate gradient
        optimizer.step()#backpropagation

        running_loss+=loss.item()
        if i%2000==1999:
            print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/2000))
            running_loss=0.0
print('Finished Trainning')

#test loop
correct=0
total=0
with torch.no_grad():#for test
    for data in testloader:
        images, labels=data#get image, correct value
        outputs=net(images)#get output
        _, predicted=torch.max(outputs.data,1)#get result******
        total+=labels.size(0)#count
        correct+=(predicted==labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%'%(100*correct/total))
