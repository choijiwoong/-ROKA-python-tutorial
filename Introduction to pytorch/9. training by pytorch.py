import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

#data setting
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])

training_set=torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set=torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

training_loader=torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, num_workers=0)#난 GPU가 없다...
validation_loader=torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=0)

classes=('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

print("Training set has {} instances".format(len(training_set)))
print("Validation set has {} instance".format(len(validation_set)))

#visualization by sanity check
import matplotlib.pyplot as plt
import numpy as np

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img=img.mean(dim=0)
    img=img/2+0.5
    npimg=img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1,2,)))

dataiter=iter(training_loader)
images, labels=dataiter.next()

img_grid=torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print(' '.join(classes[labels[j]] for j in range(4)))

#model
import torch.nn as nn
import torch.nn.functional as F

class GarmentClassifier(nn.Module):#transform of LeNet-5
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1=nn.Conv2d(1,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*4*4, 120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1, 16*4*4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x
model=GarmentClassifier()

loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#Adagrad, Adam

def train_one_epoch(epoch_index, tb_writer):
    running_loss, last_loss=0., 0.

    for i, data in enumerate(training_loader):#Get batch for training to DataLoader
        inputs, labels=data
        optimizer.zero_grad()#make gradient of optimizer to Zero
        outputs=model(inputs)#get predict
        loss=loss_fn(outputs, labels)#calculate loss
        loss.backward()#calculate back gradient
        optimizer.step()#update weights_backpropagation
        running_loss+=loss.item()

        if i%1000==999:#report loss per 1000 batch
            last_loss=running_loss/1000
            print('   batch {} loss: {}'.format(i+1, last_loss))
            tb_x=epoch_index*len(training_loader)+i+1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss=0.
    return last_loss#report loss per 1000 batch at last

#report to tensorboard
timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
writer=SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number=0

EPOCHS=5
best_vloss=1_000_000.#initial value for get best loss

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number+1))

    model.train(True)
    avg_loss=train_one_epoch(epoch_number, writer)#do trainning by train_one_epoch method
    model.train(False)

    running_vloss=0.0
    for i, vdata in enumerate(validation_loader):#check loss by validation
        vinputs, vlabels=vdata
        voutputs=model(vinputs)
        vloss=loss_fn(voutputs, vlabels)
        running_vloss+=vloss
    avg_vloss=running_vloss/(i+1)#get avg loss
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))#print train loss, validation loss

    writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch_number+1)
    writer.flush()#load loss of training & validation to tensorboard

    if avg_vloss<best_vloss:
        best_vloss=avg_vloss
        model_path='model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)#serialize
        
    epoch_number+=1

#for load
#saved_model=GarmentClassifier()
#saved_model.load_state_dict(torch.load(PATH))
