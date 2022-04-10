if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#간단한 데이터셋의 활용
import dezero
import numpy as np
from dezero.models import MLP
import dezero.optimizers as optimizers
import math
import dezero.functions as F

train_set=dezero.datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))

# 데이터 이어붙이기
train_set=dezero.datasets.Spiral()

batch_index=[0,1,2]
batch=[train_set[i] for i in batch_index]#여러 배치를 신경망에 넣기 위해선 ndarray로의 변환이 필요

x=np.array([example[0] for example in batch])#별거없고 각각 리스트로 저장
t=np.array([example[1] for example in batch])#0에 x 1에 t

print(x.shape)
print(t.shape)

# 학습 코드
max_epoch=300
batch_size=30
hidden_size=10
lr=1.0

train_set=dezero.datasets.Spiral()
model=MLP((hidden_size, 3))
optimizer=optimizers.SGD(lr).setup(model)

data_size=len(train_set)
max_iter=math.ceil(data_size/batch_size)

for epoch in range(max_epoch):
    index=np.random.permutation(data_size)
    sum_loss=0

    for i in range(max_iter):
        batch_index=index[i*batch_size:(i+1)*batch_size]
        batch=[train_set[i] for i in batch_index]
        batch_x=np.array([example[0] for example in batch])
        batch_t=np.array([example[1] for example in batch])

        y=model(batch_x)
        loss=F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss+=float(loss.data)*len(batch_t)
    avg_loss=sum_loss/data_size
    print('epoch %d, loss %.2f'%(epoch+1, avg_loss))

#(번외) transform
def f(x):
    y=x/2.0
    return y
train_set=dezero.datasets.Spiral(transform=f)#데이터 변형함수 추가(augmentation시 유용)

#(번외) transforms package(normalization...etc)
from dezero import transforms

f=transforms.Normalize(mean=0.0, std=2.0)
train_set=dezero.datasets.Spiral(transforms=f)#단순히 하나의 변형함수만 추가하는 것이 아니라

f=transforms.Compose([transforms.Normalize(mean=0.0, std=2.0), transforms.AsType(np.float64)])
#Compose로 transform함수들을 리스트로 묶어서 전달할 수도 있다.
#Compose는 애초에 인자를 list로 받고 call시 for로 다 돌린다.
