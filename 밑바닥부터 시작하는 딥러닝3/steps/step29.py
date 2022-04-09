#뉴턴방법 최적화_해당 위치에 접하는 테일러 급수 2차 근사 그래프의 극소점 위치로 옮긴다.
#기존에 x=x-af'(x)에서 x=x-f'(x)/f''(x)값을 의미한다. 즉 경사하강법의 학습률을 1/f''(x)로 바꾼 것과 같다
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def f(x):
    y=x**4-2*x**2
    return y

def gx2(x):#미분두번한거 수작업으로 구한거임
    return 12*x**2-4

x=Variable(np.array(2.0))
iters=10

for i in range(iters):#단 7회만에 최소값 도달
    print(i, x)

    y=f(x)
    x.cleargrad()
    y.backward()

    x.data-=x.grad/gx2(x.data)#grad는 f'(x)
