#복잡한 함수의 미분
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np
from dezero import Variable

#1
def sphere(x,y):
    z=x**2+y**2#dezero를 import하는 순간 __init__에 의해 연산자는 이미 오버로딩 되어있다.
    return z

x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=sphere(x,y)
z.backward()#미분가능
print(x.grad, y.grad)

#2
def matyas(x, y):
    z=0.26*(x**2+y**2)-0.48*x*y#convenient!
    return z

x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=matyas(x,y)
z.backward()
print(x.grad, y.grad)

#3
def goldstein(x,y):
    z=(1+(x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2)) * (30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return z

x=Variable(np.array(1.0))
y=Variable(np.array(1.0))
z=goldstein(x,y)
z.backward()
print(x.grad, y.grad)
