if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
"""pip install cupy
    
import cupy as cp
import numpy as np
x=cp.arange(6).reshape(2,3)
print(x)

y=x.sum(axis=1)
print(y)

n=np.array([1,2,3])
c=cp.asarray(n)#numpy->cupy
assert type(c)==cp.ndarray

c=cp.array([1,2,3])
n=cp.asnumpy(c)#cupy->numpy
assert type(n)==np.adarray

#데이터에 적합한 모듈 리턴
x=np.array([1,2,3])
xp=cp.get_array_module(x)
assert xp==np

x=cp.array([1,2,3])
xp=cp.get_array_module(x)
assert xp==cp"""

#cupy모듈 추가 후 전반적인 GPU지원화. Variable, Layer, DataLoader, Functions(np들어간거 몇개 없넴  다 xp로 치환), core

#GPU로 MNIST학습하기
import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch=5
batch_size=100

train_set=dezero.datasets.MNIST(train=True)
train_loader=DataLoader(train_set, batch_size)
model=MLP((1000, 10))
optimizer=optimizers.SGD().setup(model)

if dezero.cuda.gpu_enable:#GPU모드
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start=time.time()
    sum_loss=0

    for x, t in train_loader:
        y=model(x)
        loss=F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss+=float(loss.data)*len(t)

    elapsed_time=time.time()-start
    print('epoch: {}, loss: {:.4f}[sec]'.format(epoch+1, sum_loss/len(train_set), elapsed_time))
    
