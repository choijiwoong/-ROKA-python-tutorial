"""
대부분의 활성화 함수는 x=0주변에서 가장 강한 기울기를 가지므로 데이터를 중앙에 배치한다면(평균0) 학습속도가 보다 빨라질 수 있다.

"""
import torch
import torchvision
import torchvision.transforms as transforms

transform=transforms.Compose(#convert image to Tensor
    [transforms.ToTensor(),#img to tensor
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#modify tensor value until average=0 standard deviation=0.5

trainset=torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)#make instance of 6 animals & 4 vehicles
trainloader=torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

import matplotlib.pyplot as plt
import numpy as np

classess=('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img=img/2+0.5#unnormalize
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1,2, 0)))

dataiter=iter(trainloader)
images,labels=dataiter.next()

imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s'%classes[labels[j]] for j in range(4)))
