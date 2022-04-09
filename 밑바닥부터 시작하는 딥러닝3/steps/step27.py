#테일러 급수 미분: 함수를 다항식으로 근사하는 방식으로, 미분가능하게 간편화 할 수 있으며 항이 많아질수록 근사의 정확도가 높아진다.
#a=0일 때의 테일러 급수를 매클로린 전개라고 하며 보다 간단하다.
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dezero import Function, Variable
import numpy as np
import math

class Sin(Function):#정공법이용
    def forward(self, x):
        y=np.sin(x)
        return y

    def backward(self, gy):
        x=self.inputs[0].data
        gx=gy*np.cos(x)#공식. 사인미분은 코사인
        return gx
def sin(x):
    return Sin()(x)

x=Variable(np.array(np.pi/4))
y=sin(x)
y.backward()

print(y.data)
print(y.grad)

def my_sin(x, threshold=0.0001):#테일러 급수 a=0일때인 매클레인 전개 이용
    y=0
    for i in range(100000):
        c=(-1)**i/math.factorial(2*i+1)#분모
        t=c*x**(2*i+1)#분자
        y=y+t#누산
        if abs(t.data)<threshold:#근사치의 정밀도 조정
            break

    return y

x=Variable(np.array(np.pi/4))
y=my_sin(x)
y.backward()

print(y.data, y.grad)
