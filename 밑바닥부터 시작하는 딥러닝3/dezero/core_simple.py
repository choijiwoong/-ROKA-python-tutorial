import weakref
import numpy as np
import contextlib

class Config:#for grad mode change
    enable_backprop=True

@contextlib.contextmanager
def using_config(name, value):
    old_value=getattr(Config, name)#Config에서 인자 name의 현재 속성 get
    setattr(Config, name, value)#인자 value로 값 변경
    try:
        yield#양보하여 다른 작업부터
    finally:#영역이 닫힐 때
        setattr(Config, name, old_value)

def no_grad():#using_config에서 자주 사용하는 모드(False)를 따로 편리화
    return using_config('enable_backprop', False)


class Variable:
    __array_priority__=200#ndarray와의 연산 시 ndarray의 메서드가 호출되지 않게 우선순위 조정
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):#ndarray입력만 받는다
                raise TypeError('{} is not supported.'.format(type(data)))

        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0#완벽한 미분을 위함(repeat의 역전파)

    @property#유용한 메서드 제공. shape(var)로 사용
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):#유용한 값 제공. var.len으로 사용
        return len(self.data)

    def __repr__(self):#print(var)시 리턴값
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n'+' '*9)
        return 'variable('+p+')'

    def set_creator(self, func):
        self.creator=func
        self.generation=func.generation+1

    def cleargrad(self):
        self.grad=None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_set=set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]#output은 순환참조를 막기 위해 weakref이므로 ()로 사용한다.
            gxs=f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs=(gxs,)

            for x, gx in zip(f.inputs, gxs):#미분값 갱신 및 새 creator 추가
                if x.grad is None:
                    x.grad=gx
                else:
                    x.grad=x.grad+gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:#불필요한 미분값 삭제
                for y in f.outputs:
                    y().grad=None


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x#스칼라가 아니면 그냥 반환.


class Function:
    def __call__(self, *inputs):#variadic argument
        inputs=[as_variable(x) for x in inputs]

        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:#미분을 위한 정보가 필요하다면
            self.generation=max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs=inputs
            self.outputs=[weakref.ref(output) for output in outputs]#output을 weakref로 저장(미분필요시)

        return outputs if len(outputs)>1 else outputs[0]#원소하나면 그거만

    def forward(self, xs):#추상클래스
        raise NotImplementedError()
    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y

    def backward(self, gy):
        return gy, gy
def add(x0, x1):
    x1=as_array(x1)#더해지는 값이 scalar일수도 있으니
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y=x0*x1
        return y

    def backward(self, gy):
        x0, x1=self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0
def mul(x0, x1):
    x1=as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):#(negation은 미분도 negation!)
        return -gy
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y=x0-x1
        return y

    def backward(self, gy):
        return gy, -gy
def sub(x0, x1):
    x1=as_array(x1)
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1=as_array(x1)
    return sub(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y=x0/x1
        return y

    def backward(self, gy):
        x0, x1=self.inputs[0].data, self.inputs[1].data
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)#분모 편미분하면 -1/x^2
        return gx0, gx1
def div(x0, x1):
    x1=as_array(x1)
    return Div()(x0,x1)
def rdiv(x0, x1):
    x1=as_array(x1)
    return div(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c=c

    def forward(self, x):
        y=x**self.c
        return y

    def backward(self, gy):
        x=self.inputs[0].data
        c=self.c

        gx=c*x**(c-1)*gy
        return gx
def pow(x,c):
    return Pow(c)(x)


def setup_variable():#연산자 오버로딩 처리
    Variable.__add__=add
    Variable.__radd__=add
    Variable.__mul__=mul
    Variable.__rmul__=mul
    Variable.__neg__=neg
    Variable.__sub__=sub
    Variable.__rsub__=rsub
    Variable.__truediv__=div
    Variable.__rtruediv__=rdiv
    Variable.__pow__=pow
