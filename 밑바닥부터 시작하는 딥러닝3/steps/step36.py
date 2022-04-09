if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#고차 미분의 사용1_변수가 겹치는 두 식의 미분 계산.(치환느낌, 편미분갬성)
import numpy as np
from dezero import Variable

x=Varaible(np.array(2.0))
y=x**2
y.backward(create_graph=True)
gx=x.grad#미분값을
x.cleargrad()

z=gx**2+y#식에사용하고
z.backward()#그 식을 역전파함으로서 미분값이 사용된 식마저 미분해버릴 수 있다.
print(x.grad)
