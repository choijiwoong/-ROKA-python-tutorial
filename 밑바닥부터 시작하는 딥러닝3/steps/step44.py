if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Parameter
import dezero.layers as L
import dezero.functions as F

x=Variable(np.array(1.0))
p=Parameter(np.array(2.0))
y=x*p
print(isinstance(p, Parameter))#구분가능
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))#업캐스팅


layer=L.Layer()
layer.p1=Parameter(np.array(1))#__setattr__호출
layer.p2=Parameter(np.array(2))
layer.p3=Variable(np.array(3))
layer.p4='test'

print(layer._params)

for name in layer._params:
    print(name, layer.__dict__[name])#__dict__에는 모든 인스턴스 변수가 딕셔너리로 저장되어있다.


np.random.seed(0)#Layer을 이용한 신경망 구현
x=np.random.rand(100,1)
y=np.sin(2*np.pi*x)+np.random.rand(100,1)

l1=L.Linear(10)#출력크기
l2=L.Linear(1)

def predict(x):
    y=l1(x)
    y=F.sigmoid(y)
    y=l2(y)
    return y

lr=0.2
iters=10000

for i in range(iters):
    y_pred=predict(x)
    loss=F.mean_squared_error(y, y_pred)
    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data-=lr*p.grad.data
    if i%1000==0:
        print(loss)
