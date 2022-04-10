if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero.functions as F
from dezero import Variable

# 형상의 변환
x=np.array([[1,2,3], [4,5,6]])#numpy의 reshape예시
y=np.reshape(x, (6,))
print(y)

x=Variable(np.array([[1,2,3], [4,5,6]]))
y=F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)#원래의 형상으로 돌아온다! grad와 원래 shape랑 동일

#Variable에 편하게 적용하기 위해 클래스에 추가해주자.
x=Variable(np.random.randn(1,2,3))
y=x.reshape((2,3))#튜플, 리스트, 혹은 풀어서 reshapeing가능. 내부적으로 np의 reshape를 사용하기에
y.x.reshape(2,3)

# 행렬의 전치
x=np.array([[1,2,3], [4,5,6]])#numpy 예시
y=np.transpose(x)
print(y)

x=Variable(np.array([[1,2,3], [4,5,6]]))
y=F.transpose(x)#마찬가지로 Variable에 추가.
y.backward()
print(x.grad)#(실제 transpose는 axes인자로 바꿀 축을 선택할 수 있기에 반영해둔다.)
