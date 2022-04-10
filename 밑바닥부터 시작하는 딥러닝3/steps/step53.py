if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

x=np.array([1,2,3])#1. numpy의 저장로드지원(단일)
np.save('test.npy', x)#(np.savez_compressed로 압출하여 저장할수도 있다)

x=np.load('test.npy')
print(x)

x1=np.array([1,2,3])#2. numpy의 저장로드지원(다수)
x2=np.array([4,5,6])
#data={'x1':x1, 'x2':x2}
#np.savez('test.npz', **data) 처럼 직접 dict를 전달해 savez에서 unpack하여 저장할 수도 있음

np.savez('test.npz', x1=x1, x2=x2)#dict형태로 저장

arrays=np.load('test.npz')
x1=arrays['x1']
x2=arrays['x2']
print(x1)
print(x2)


#Layer클래스의 parameter들을 flatten해야 저장이 가능하니 메서드 추가.(현재 layer가 layer을 품을 수 있게 되어있기에)
import os
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

max_epoch=3
batch_size=100

train_set=dezero.datasets.MNIST(train=True)
train_laoder=DataLoader(train_set, batch_size)
model=MLP((1000, 10))
optimizer=optimizers.SGD().setup(model)

if os.path.exists('mp_mlp.npz'):#매개변수 파일 읽기(기존에 학습시킨게 있다면)
    model.load_weights('my_mlp.npz')
for epoch in range(max_epoch):
    sum_loss=0

    for x,t in train_loader:
        y=model(x)
        loss=F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss+=float(loss.data)*len(t)
    print('epoch: {}, loss: {:.4f}'.format(epoch+1, sum_loss/len(train_set)))
model.save_weights('my_mlp.npz')
