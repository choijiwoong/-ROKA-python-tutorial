#리펙토링!
import torch.nn.functional as F#for cross_entropy_lof softmax+nll
from torch import nn#nn.Module for make Mnist_ligistic for reducing class model~
from torch import optim#for using optimizer
from torch.utils.data import TensorDataset#for entering x_train, y_train to Dataset
#등을 이용하여 각종 리팩터링을 했지만 너무 예시가 난잡하여 부분적으로 참고할 수 있는 부분만 정리해두겠음.

#[fit(), get_data()]
import numpy as np

def loss_batch(model, loss_func, xb, yb, opt=None):#Calculate loss of one batch
    loss=loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, tarin_dl, valid_dl):
    for epoch in range(epochs):
        model.train()#training
        for xb, yb in train_dl:#execute loss_batch per each batches
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()#evaluation
        with torch.no_grad():
            losses, nums=zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]#Comprehension!
            )
        val_loss=np.sum(np.multiply(losses, nums))/np.sum(nums)#just loss

        print(epoch, val_loss)
        
def get_data(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(valid_ds, batch_size=bs*2), )

#[CNN]
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2=nn.Conv2d(16,16, kernel_size=3, stride=2, padding=1)
        self.conv3=nn.Conv2d(16,10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb=xb.view(-1, 1, 28, 29)
        xb=F.relu(self.conv1(xb))
        xb=F.relu(self.conv2(xb))
        xb=F.relu(self.conv3(xb))
        xb=F.age_pool2d(xb, 4)
        return wb.view(-1, xb.size(1))
lr=0.1

model=Mnist_CNN()
opt=optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#[nn.Sequential]
class Lambda(nn.Module):#사용자정의 레이어를 쉽게 정의할 수 있어야 한다..
    def __init__(self, func):
        super().__init__()
        self.func=func

    def forward(self, x):
        return self.func(x)

def preprocess(x):
    return x.view(-1,1,28,28)

model=nn.Sequential(
    Lambda(preprocess),#User defined layer. it can forward(now, set self.func to preprocess) DL을 view를통해 평탄화하거나
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt=optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

#[DataLoader Wrapper]
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl=dl#dataloader
        self.func=func#function

    def __len__(self):
        return len(self.dl)#return size of dataloader

    def __iter__(self):
        batches=iter(self.dl)#iteraterize dataloader
        for b in batches:#for all batches in dataloader
            yield(self.func(*b))#can approach by iterator
train_dl, valid_dl=get_data(train_ds, valid_ds, bs)
train_dl=WrappedDataLoader(train_dl, preprocess)
valid_dl=WrappedDataLoader(valid_dl, preprocess)

model=nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvdPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt=optim.SGD(model.parameters(), lr=lr, momentum=0.9)
fit(epoches, model, loss_func, opt, train_dl, valid_dl)#iterator이용하여 개별 접근 가능하게 하거나
