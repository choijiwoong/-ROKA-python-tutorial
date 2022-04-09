#뉴턴방법으로 푸는 최적화
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def f(x):
    y=x**4-2*x**2
    return y

x=Variable(np.array(2.0))
y=f(x)
y.backward(create_graph=True)#한번더 미분할거임
print(x.grad)#24

gx=x.grad
x.cleargrad()#미분값을 리셋하지 않으면 같은 변수를 통해 찾기에 미분값이 누산된다.
gx.backward()
print(x.grad)#44

#뉴턴식 최적화
x=Variable(np.array(2.0))
iters=10

for i in range(iters):
    print(i, x)

    y=f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx=x.grad
    x.cleargrad()#2차미분을 위한 리셋
    gx.backward()
    gx2=x.grad

    x.data-=gx.data/gx2.data#SGD의 learning rate를 1/f''(x)로. 뉴턴식 최적화!
