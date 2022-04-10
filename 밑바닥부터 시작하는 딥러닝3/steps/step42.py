if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#선형회귀에 쓰일 토이 데이터셋
import numpy as np

np.random.seed(0)
x=np.random.rand(100, 1)
y=5+2*x+np.random.rand(100,1)#무작위 노이즈(선형회귀 적용할 데이터)

#구현
from dezero import Variable
import dezero.functions as F

x, y=Variable(x), Variable(y)

W=Variable(np.zeros((1,1)))
b=Variable(np.zeros(1))

def predict(x):
    y=F.matmul(x,W)+b
    return y

def mean_squared_error(x0, x1):#간단한 구현. 이대로 사용하면 사용된 변수들이 메모리에 남게 된다. 고로 Function상속 클래스화
    diff=x0-x1
    return F.sum(diff**2)/len(diff)
lr=0.1
iters=100

for i in range(iters):
    y_pred=predict(x)
    loss=F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data-=lr*W.grad.data
    b.data-=lr*b.grad.data
    print(W,b,loss)
