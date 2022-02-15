import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),#convert PIL image or ndarray to FloatTensor, scale intensity in range[0,1]
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    #if correct, make it's index to 1 or 0 by default value.
)
