from step03 import *

def numerical_diff(f, x, eps=1e-4):#centered differenctiation이용. not forward difference
    x0=Variable(x.data-eps)
    x1=Variable(x.data+eps)
    y0=f(x0)
    y1=f(x1)
    return (y1.data-y0.data)/(2*eps)#자리수 누락, 많은 계산량이 단점. 고로 버그가능성이 높은역전파 시 gradient checking으로 사용

"""1. 일반함수 미분
f=Square()
x=Variable(np.array(2.0))
dy=numerical_diff(f, x)
print(dy)

#2. 합성함수 미분
def f(x):
    A=Square()
    B=Exp()
    C=Square()
    return C(B(A(x)))

x=Variable(np.array(0.5))
dy=numerical_diff(f,x)
print(dy)"""
