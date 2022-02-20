#PyTorch model & training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Image dataset & image manipulation
import torchvision
import torchvision.transforms as transforms

#Image display
import matplotlib.pyplot as plt
import numpy as np

#Pytorch Tensorboard support
from torch.utils.tensorboard import SummaryWriter

transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

training_set=torchvision.datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
validation_set=torchvision.datasets.FashionMNIST('/data', download=True, train=False, transform=transform)
training_loader=torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
validation_loader=torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)

#Class Labels
classes=("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot")

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        if one_channel:
            img=img.mean(dim=0)
        img=img/2+0.5#unnormalize
        npimg=img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1,2,0)))#2nd arg축을 기준으로 열과행변환
#Extract a batch of 4 images
dataiter=iter(training_loader)
images, labels=dataiter.next()

#Create grid and Show
img_grid=torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)

#default lof for log is 'runs'
writer=SummaryWriter('runs/fashion_mnist_experiment_1')
#write img data to TensorBoard log dir
writer.add_image('Four Fasion-MNIST Images', img_grid)
writer.flush()#보기위해서는 tensorboard --logdir=runs를 cmd에 입력하고 http://localhost:6006에 접속하면 된다.


#[Graph scalar for visualization of training]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1=nn.Conv2d(1,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*4*4, 120)
        self.fc2=nn.Linear(120, 84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1, 16*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
net=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#do train & test by once
print('validation_loader length: ',len(validation_loader))
for epoch in range(1):
    running_loss=0.0

    for i, data in enumerate(training_loader, 0):
        inputs, labels=data
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        
        if i%1000==999:
            print('Batch {}'.format(i+1))
            running_vloss=0.0

            net.train(False)#test per 1000train
            for j, vdata in enumerate(validation_loader, 0):#validation part!
                vinputs, vlabels=vdata
                voutputs=net(vinputs)
                vloss=criterion(voutputs, vlabels)
                running_vloss+=vloss.item()
            net.train(True)

            avg_loss=running_loss/1000
            avg_vloss=running_vloss/len(validation_loader)

            writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch*len(training_loader)+i)
            #add loss result to tensorboard per epoch 1000
            running_loss=0.0
print('Finished Training')
writer.flush()

    
#[Model Visualization]#tensorboard graph탭의 net을 더블클릭하여 graph의 흐름을 가시화가능
dataiter=iter(training_loader)
images, labels=dataiter.next()

writer.add_graph(net, images)#check flow of data in model
writer.flush()


#[Dataset Visualization by embedding]_Project탭으로 투영의 3D표현가능.. tensorboard_writer의 add_embedding
#select a random subset of data ans corresponding labels
def select_n_random(data, labels, n=100):
    assert len(data)==len(labels)

    perm=torch.randperm(len(data))#shuffle
    return data[perm][:n], labels[perm][:n]

#extract a random subset of data
images, labels=select_n_random(training_set.data, training_set.targets)
class_labels=[classes[label] for label in labels]#get class labels

features=images.view(-1,28*28)
writer.add_embedding(features, metadata=class_labels, label_img=images.unsqueeze(1))
writer.flush()
writer.close()
