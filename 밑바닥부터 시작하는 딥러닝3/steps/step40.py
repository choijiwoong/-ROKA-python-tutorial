#브로드캐스팅 함수
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero.core import Variable

x=np.array([1,2,3])
y=np.broadcast_to(x, (2,3))#numpy예시
print(y)

from dezero.utils import sum_to

x=np.array([[1,2,3], [4,5,6]])
y=sum_to(x, (1,3))#(구현예시)브로드캐스팅의 반대
print(y)

y=sum_to(x, (2,1))
print(y)

#브로드캐스트 대응
x0=np.array([1,2,3])
x1=np.array([10])
y=x0+x1
print(y)

x0=Variable(np.array([1,2,3]))
x1=Variable(np.array([10]))
y=x0+x1#Add클래스의 shape가 맞지 않을 때를 추가해야함.
print(y)

#모든 사칙연산 클래스에서 broadcast를 반영한 backward로 수정.
x0=Variable(np.array([1,2,3]))
x1=Variable(np.array([10]))
y=x0+x1
print(y)

y.backward()
print(x1.grad)
