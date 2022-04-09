#함수 최적화(경사하강법)_기울기는 y값을 가장 크기 혹은 작게 해주는 방향 자체로 해석할 수도 있다.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y=100*(x1-x0**2)**2+(1-x0)**2
    return y

x0=Variable(np.array(0.0))
x1=Variable(np.array(2.0))
lr=0.001
iters=1000

for i in range(iters):
    print(x0, x1)

    y=rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data-=lr*x0.grad
    x1.data-=lr*x1.grad
