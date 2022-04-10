if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# 슬라이싱 함수
import numpy as np
from dezero import Variable, as_variable
import dezero.functions as F

x=Variable(np.array([[1,2,3], [4,5,6]]))
y=F.get_item(x, 1)#슬라이싱 함수
print(y)#[4,5,6]
           
y.backward()
print(x.grad)#미분가능한.

indices=np.array([0,0,1])
y=F.get_item(x, indices)
print(y)#인덱스 반복지정하여 동일원소 여러번 추출도 가능한.

#Variable __getitem__ 특수메서드로 설정(setup_variable).
y=x[1]
print(y)

y=x[:,2]
print(y)

# 소프트맥스
from dezero.models import MLP

model=MLP((10,3))

def softmax1d(x):
    x=as_variable(x)
    y=F.exp(x)
    sum_y=F.sum(y)#exp화 한 뒤 비율
    return y/sum_y
x=np.array([[0.2, -0.4]])
y=model(x)
p=softmax1d(y)
print(y)
print(p)

# 교차 엔트로피 오차
x=np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t=np.array([2,0,1,0])#x가 모델을 통해 나와야하는 값
y=model(x)
loss=F.softmax_cross_entropy_simple(y, t)
print(loss)
