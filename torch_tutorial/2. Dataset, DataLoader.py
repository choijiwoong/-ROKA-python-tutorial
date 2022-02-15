#readability, modularity를 위해 데이터셋코드(Dataset_sample, answer), 모델 학습 코드(dataloader_iterable object for access)로 분리한다.
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#[1. get MNIST dataset]
traning_data=datasets.FashionMNIST(
    root="data",#save path of data
    train=True,
    download=True,#if non-data, download to internet automatically
    transform=ToTensor()#target_transform도 있음
)
test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#recursive & virtualize dataset
labels_map = {#labels_map for approaching FasionMNIST dataset(need label for approach)
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure=plt.figure(figsize=(8,8))#make figure with size
cols, rows=3,3
for i in range(1, cols*rows+1):
    sample_idx=torch.randint(len(training_data), size=(1,)).item()#get any index of training_data as int type
    img, label=traning_data[sample_idx]#get sample data to img, label variable
    figure.add_subplot(rows, cols, i)#add subplot
    plt.title(labels_map[label])#get label(title) to sub_plot's title
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")#squeeze remove dimention0
plt.show()

#[2. customized dataset_based on local csv file]
import os
import pandas as pd#for read csv
from torchvision.io import read_image

class CustomImageDataset(Dataset):#inheritate Dataset
    def __init__(self, annotaions_file, img_dir, transform=None, target_transform=None):#get csv dir, img dir
        self.img_labels=pd.read_csv(annotations_file, names=['file_name', 'label'])#get csv
        self.img_dir=img_dir
        self.transform=transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self,img_labels)

    def __getitem__(delf, idx):#get item by index
        img_path=os.path.join(self.img_dir, self.img_labels.iloc[idx,0])#get file by index
        image=read_image(img_path)#image to tensor
        label=self.img_labels.iloc[idx,1]
        if self.transform:#global modifing function_image
            image=self.transform(image)
        if self.target_transform:#local modifing function_label
            label=self.target_transform(label)
        sample={"image": image, "label": label}
        return sample#as dictionary

#[3. recursive by dataloader]
from torch.utils.data import DataLoader

train_dataloader=DataLoader(training_data, batch_size=64, shuffle=True)#ready dataloader
test_dataloader=DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels=next(iter(train_dataloader))#return dictionary_리턴된 데이터 처리하는거 잘 이해안됨
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img=train_feature[0].squeeze()#zip dimention
label=train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
