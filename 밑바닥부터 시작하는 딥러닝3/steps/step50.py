if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

#반복자: 데이터타입으로부터 데이터를 순차적으로 추출하는 기능. iter(), next(), raise StopIteration()
class MyIterator:#간단한 구현 예시
    def __init__(self, max_cnt):
        self.max_cnt=max_cnt
        self.cnt=0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt==self.max_cnt:
            raise StopIteration()

        self.cnt+=1
        return self.cnt
    
obj=MyIterator(5)
for x in obj:
    print(x)

#데이터 로더도 위의 반복자의 개념으로 next시 다음 데이터를 로드하게끔 구현한다.
from dezero.datasets import Spiral
from dezero import DataLoader

batch_size=10
max_epoch=1

train_set=Spiral(train=True)
test_set=Spiral(train=False)
train_loader=DataLoader(train_set, batch_size)
test_loader=DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break

    for x, t in test_loader:
        print(x.shape, t.shape)
        break

#accuracy함수 예시
import dezero.functions as F
import numpy as np

y=np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
t=np.array([1,2,0])
acc=F.accuracy(y,t)
print('\n',acc,'\n')

#스파이럴 데이터셋 학습 코드(feat. 데이터 로더)
import dezero
from dezero.models import MLP
import dezero.optimizers as optimizer

max_epoch=300
batch_size=30
hidden_size=10
lr=1.0

train_set=dezero.datasets.Spiral(train=True)
test_set=dezero.datasets.Spiral(train=False)
train_loader=DataLoader(train_set, batch_size)
test_laoder=DataLoader(test_set, batch_size)

model=MLP((hidden_size, 3))
optimizer=optimizer.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc=0, 0

    for x, t in train_loader:
        y=model(x)
        loss=F.softmax_cross_entropy(y,t)
        acc=F.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)
    print('epoch: ',epoch+1)
    print('train_loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(train_set), sum_acc/len(train_set)))

    sum_loss, sum_acc=0,0
    with dezero.no_grad():
        for x, t in test_loader:
            y=model(x)
            loss=F.softmax_cross_entropy(y,t)
            acc=F.accuracy(y,t)
            sum_loss+=float(loss.data)*len(t)
            sum_acc+=float(acc.data)*len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), sum_acc/len(test_set)))
    
