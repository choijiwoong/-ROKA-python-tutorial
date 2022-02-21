#[Prepare MNIST data]
from pathlib import Path
import requests

DATA_PATH=Path('data')
PATH=DATA_PATH/"mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL="https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME="mnist.pkl.gz"#download to github

if not (PATH/FILENAME).exists():
    content=requests.get(URL+FILENAME).content
    (PATH/FILENAME).open("wb").write(content)#make file if it's not exist

#load serialized data
import pickle
import gzip

with gzip.open((PATH/FILENAME).as_posix(), "rb") as f:#with-resource
    ((x_train, y_train), (x_valid, y_valid), _)=pickle.load(f, encoding="latin-1")

#show data
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28,28)), cmap="gray")
print("x_train.shape we download: ", x_train.shape,end='\n\n')

#we will use tensor
import torch

x_train, y_train, x_valid, y_valid=map(torch.tensor, (x_train, y_train, x_valid, y_valid))#not STL container's map. mapping to torch.tensor
n, c=x_train.shape#n!!!!
print("x_train:", x_train, "y_train:", y_train)
print("shape of x_train: ",x_train.shape)
print("y_train.min(): ", y_train.min(), "y_train.max(): ", y_train.max(), end='\n\n')

#[Make Neural Network without torch.nn]
import math

weights=torch.randn(784, 10)/math.sqrt(784)#가중치 초기화 기법 중 하나로 Xavier Initialization이다. 이전 노드와 다음 노드의 개수에 의존하는 방법
weights.requires_grad_()
bias=torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x-x.exp().sum(-1).log().unsqueeze(-1)#log(x-sum(e^x)_softmax). x is matrix(one data) because we will get multi data as much as batch_size
def model(xb):
    return log_softmax(xb @ weights+bias)#matrix multiplication

batch_size=64

xb=x_train[0:batch_size]#extract mini-batch
preds=model(xb)
print("preds[0]: ",preds[0], "shape of preds: ", preds.shape, end='\n\n')

#define loss function
def nll(input, target):#Negative Log-Likelihood as loss function
    return -input[range(target.shape[0]), target].mean()
loss_func=nll

yb=y_train[0:batch_size]
print("loss with model(x_train), y_train: ", loss_func(preds, yb))#check loss 여기서 질문인게 다운로드한 타입의 x와 y가 뭔지 모르겠네. 모델통과시킨 x예측이랑 y train이랑 손실을 계산한다고..? 혹시 x_train을 모델통과시킨 예측이 y_train이고, x_valid의 정답이 y_valid인건가

#calculate accuracy
def accuracy(out, yb):
    preds=torch.argmax(out, dim=1)#max of predict
    return (preds==yb).float().mean()#get average of perfectly correct
print("accuracy(preds, yb): ",accuracy(preds, yb))


#training loop
from IPython.core.debugger import set_trace

lr=0.5
epochs=2

for epoch in range(epochs):
    for i in range((n-1)//batch_size+1):#maybe length of data
        #set_trace()
        start_i=i*batch_size
        end_i=start_i+batch_size

        xb=x_train[start_i:end_i]
        yb=y_train[start_i:end_i]

        pred=model(xb)
        loss=loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights-=weights.grad*lr#use GCD
            bias-=bias.grad*lr
            weights.grad.zero_()#make it to zero
            bias.grad.zero_()
print("loss_func(model(xb), yb)", loss_func(model(xb), yb), "accuracy(model(xb), yb): ", accuracy(model(xb), yb))
