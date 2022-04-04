import numpy as np#전반적인 사용 개선

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):#3) ndarray만 수용한다
                raise TypeError("{} is not supported".format(type(data)))
        self.data=data
        self.grad=None
        self.creator=None

    def set_creator(self, func):
        self.creator=func

    def backward(self):
        if self.grad is None:#2) 기존의 self.grad=np.array(1.0)를 조건부로 변경
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            x,y=f.input, f.output
            x.grad=f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

def as_array(x):
    if np.isscalar(x):#numpy특성상 제곱을 하면 float32가 되는 특성이 있기에 scalar경우만 따로 처리
        return np.array(x)
    return x

class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(as_array(y))#4) 위와 같은 이유로 array로 forward후 ndarray로 바꿔줌.
        output.set_creator(self)
        self.input=input
        self.output=output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, hy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y=x**2
        return y

    def backward(self, gy):
        x=self.input.data
        gx=2*x*gy
        return gx

class Exp(Function):
    def forward(self, x):
        y=np.exp(x)
        return y

    def backward(self, gy):
        x=self.input.data
        gx=np.exp(x)*gy
        return gx

def square(x):#1) 파이썬 함수 지원
    return Square()(x)

def exp(x):
    return Exp()(x)

x=Variable(np.array(0.5))
y=square(exp(square(x)))
y.backward()
print(x.grad)

x=Variable(np.array(1.0))#OK
x=Variable(None)#OK
x=Variable(1.0)#TypeError
