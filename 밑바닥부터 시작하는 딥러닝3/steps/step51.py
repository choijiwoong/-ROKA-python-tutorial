if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# MNIST 데이터셋
import dezero

train_set=dezero.datasets.MNIST(train=True, transform=None)
test_set=dezero.datasets.MNIST(train=False, transform=None)

print(len(train_set))#데이터 가져오기
print(len(test_set))

x, t=train_set[0]
print(type(x), x.shape)
print(t)#샘플데이터 확인

import matplotlib.pyplot as plt

x, t=train_set[0]
plt.imshow(x.reshape(28,28), cmap='gray')
plt.axis('off')
plt.show()
print('label: ', t)

# MNIST 학습하기
"""
def f(x):#전처리 (hooks)
    x=x.flatten()
    x=x.astype(np.float32)
    x/=255.0
    return x

train_set=dezero.datasets.MNIST(train=True, transform=f)
test_set=dezero.datasets.MNIST(train=False, transform=f)
#으로 하면 정석이지만 MNIST는 이미 이러한 전처리를 제공해준다.
"""
from dezero.models import MLP
import dezero.optimizers as optimizers
import dezero.functions as F
max_epoch=5
batch_size=100
hidden_size=1000

train_set=dezero.datasets.MNIST(train=True)
test_set=dezero.datasets.MNIST(train=False)
train_loader=DataLoader(train_set, batch_size)
test_loader=DataLoader(test_set, batch_size, shuffle=False)

model=MLP((hidden_size, 10), activation=F.relu)
optimizer=optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    sum_loss ,sum_acc=0,0
    for x, t in train_laoder:
        y=model(x)
        loss=F.softmax_cross_entropy(y,t)
        acc=F.accuracy(y,t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss+=float(loss.data)*len(t)
        sum_acc+=float(acc.data)*len(t)

    print('epoch: ', epoch+1)
    print('train loss: {:.4}, accuracy: {:.4f}'.format(sum_loss/len(train_set), sum_acc/len(train_set)))

    sum_loss, sum_axx=0,0
    with dezero.no_grad():
        for x, t in test_loader:
            y=model(x)
            loss=F.softmax_cross_entropy(y, t)
            acc=F.accuracy(y, t)
            sum_loss+=float(loss.data)*len(t)
            sum_acc+=float(acc.data)*len(t)
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss/len(test_set), aum_acc/len(test_set)))
