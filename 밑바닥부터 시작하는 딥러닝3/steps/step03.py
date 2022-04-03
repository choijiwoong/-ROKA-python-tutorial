from step02 import *

class Exp(Function):
    def forward(self, x):
        return np.exp(x)#y=e^x
"""
A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)#Composite function(합성함수)

print(y.data)"""
