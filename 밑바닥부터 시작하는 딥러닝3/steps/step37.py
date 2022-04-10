#텐서를 다루다
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
from dezero import Variable

x=Variable(np.array(1.0))#원소 1개
y=F.sin(x)
print(y)

x=Variable(np.array([[1,2,3], [4,5,6]]))#원소 여러개
y=F.sin(x)
print(y)#element-wise

x=Variable(np.array([[1,2,3], [4,5,6]]))
c=Variable(np.array([[10,20,30], [40,50,60]]))
y=x+c
print(y)#element-wise

y=F.sum(y)
print(y)#scalar(텐서의 모든 원소 총합을 계산)

y.backward(retain_grad=True)
print(y.grad)
print(c.grad)
print(x.grad)#기울기의 형상과 데이터의 형상이 일치한다. x.shape==x.grad.shape

#(p.s) 명시적 야코비 행렬로 곱하기보다 결과만 구하면 되기에 원소별 곱해지는 값만(대각행렬)구해서 곱해도 된단다..
