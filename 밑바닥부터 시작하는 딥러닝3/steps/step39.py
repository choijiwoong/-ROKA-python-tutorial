#sum함수인데 벡터에서의 sum의 미분은 원소만큼 복사(broadcast)해야하는데 이 작업도(복사) Variable에 적용되기에 Function으로 구현해야한다.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
from dezero import Variable

x=Variable(np.array([1,2,3,4,5,6]))
y=F.sum(x)#원소별 합
y.backward()#본래의 형상에 맞게 gy복사
print(y)
print(x.grad)

x=Variable(np.array([[1,2,3], [4,5,6]]))#단순한 벡터 외에도 grad시 정상 크기로 복구 by broadcast_to()
y=F.sum(x)
y.backward()
print(y)
print(x.grad)

# numpy는 섬세하다
x=np.array([[1,2,3], [4,5,6]])
y=np.sum(x, axis=0)#numpy의 sum 축 지정
print(y)
print(x.shape, '->', y.shape)

x=np.array([[1,2,3], [4,5,6]])
y=np.sum(x, keepdims=True)#numpy의 keepdims(입출력 차원의 수를 똑같이 유지할 지, scalar로 만들지)
print(y)
print(y.shape)

#Variable에도 추가
x=Variable(np.array([[1,2,3], [4,5,6]]))
y=F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x=Variable(np.random.randn(2,3,4,5))
y=x.sum(keepdims=True)
print(y.shape)
