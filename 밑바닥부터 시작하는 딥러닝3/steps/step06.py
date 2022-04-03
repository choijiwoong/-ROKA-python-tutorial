import numpy as np

class Variable:
    def __init__(self, data):
        self.data=data
        self.grad=None

class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        
        self.input=input#나중의 backward를 위해서 해당 미분값을 계산하는데에 필요하기에 저장한다.cache
        
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y=x**2#y=x^2
        return y

    def backward(self, gy):
        x=self.input.data#Function의 init시 input으로 들어온 Variable값의 data를 get
        gx=2*x*gy#y'=2x
        return gx

class Exp(Function):
    def forward(self, x):
        y=np.exp(x)#y=e^x
        return y

    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy#y'=e^x
        return gx
"""
A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)#순전파

y.grad=np.array(1.0)#자기자신의 미분값은 1
b.grad=C.backward(y.grad)
a.grad=B.backward(b.grad)
x.grad=A.backward(a.grad)
print(x.grad)#역전파 결과dx"""
