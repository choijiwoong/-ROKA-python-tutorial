#고차미분가능버전
import weakref
import numpy as np
import contextlib
import dezero
from dezero import cuda

class Config:
    enable_backprop=True
    train=True

@contextlib.contextmanager
def using_config(name, value):
    old_value=getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def test_mode():#for dropout
    return using_config('train', False)

def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    __array_priority__=200
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported.'.format(type(data)))

        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0

    @property
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

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n'+' '*9)
        return 'variable('+p+')'

    def set_creator(self, func):
        self.creator=func
        self.generation=func.generation+1

    def cleargrad(self):
        self.grad=None

    def backward(self, retain_grad=False, create_graph=False):#마찬가지로 create_grad로 역전파의 역전파 활성화모드를 조절한다.
        if self.grad is None:#단순하게 Variable로 감싸므로서 연산을 순전파와 마찬가지로 저장하고 미분할 수 있게 됨
            self.grad=Variable(np.ones_like(self.data))#np.ones_like(self.data)

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
            gys=[output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):#backward시 추가적인 backward가 필요없다면 backward과정을 다 기록하지 않게 한다.
                gxs=f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs=(gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad=gx
                    else:
                        x.grad=x.grad+gx
    
                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad=None

    def reshape(self, *shape):
        if len(shape)==1 and isinstance(shape[0], (tuple, list)):
            shape=shape[0]#원소 하나만 뾱(만약 (3,)이런식이면 (3,1)로 reshape할 수 있으니..)
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes):#function꺼 호출전에 buffer느낌인데 약간의 전처리 포함.
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self):#self.data를 numpy로
        if self.data is not None:
            self.data=dezero.cuda.as_numpy(self.data)

    def to_gpu(self):#self.data를 cupy로
        if self.data is not None:
            self.data=dezero.cuda.as_cupy(self.data)

    def unchain(self):#RNN BPTT
        self.creator=None

    def unchain_backward(self):#모든 변수의 연결 타고가며 끊기.(세대 고려X)
        if self.creator is not None:
            funcs=[self.creator]
            while funcs:
                f=funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()
        

class Parameter(Variable):#for 구분
    pass

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x, array_module=np):
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        inputs=[as_variable(x) for x in inputs]

        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation=max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs=inputs
            self.outputs=[weakref.ref(output) for output in outputs]

        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape=x0.shape, x1.shape#broadcast시 미분을 위해 저장
        y=x0+x1
        return y

    def backward(self, gy):
        gx0, gx1=gy, gy
        if self.x0_shape!=self.x1_shape:#초기 인자로 들어온 둘의 형상이 달랐다면(x0+x1시 np에 의해 브로드캐스팅되었다면)
            gx0=dezero.functions.sum_to(gx0, self.x0_shape)#본래의 형상으로 sum_to(broadcast된 만큼 누산)
            gx1=dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
def add(x0, x1):
    x1=as_array(x1, cuda.get_array_module(x0.data))
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape=x0.shape, x1.shape
        y=x0*x1
        return y

    def backward(self, gy):
        x0, x1=self.inputs
        gx0, gx1=gy*x1, gy*x0
        if self.x0_shape!=self.x1_shape:
            gx0=dezero.functions.sum_to(gx0, self.x0_shape)
            gx1=dezero.functions.sum_to(gx1, self.x1_shape)
        #Variable로 저장되니(not ndarray). 나머지도 ndarray로 받으면 변경.
        return gx0, gx1
def mul(x0, x1):
    x1=as_array(x1, cuda.get_array_module(x0.data))
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape=x0.shape, x1.shape
        y=x0-x1
        return y

    def backward(self, gy):
        gx0, gx1=gy, -gy
        if self.x0_shape!=self.x1_shape:
            gx0=dezero.functions.sum_to(gx0, self.x0_shape)
            gx1=dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
def sub(x0, x1):
    x1=as_array(x1, cuda.get_array_module(x0.data))
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1=as_array(x1, cuda.get_array_module(x0.data))
    return sub(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape=x0.shape, x1.shape
        y=x0/x1
        return y

    def backward(self, gy):
        x0, x1=self.inputs
        gx0=gy/x1
        gx1=gy*(-x0/x1**2)
        if self.x0_shape!=self.x1_shape:
            gx0=dezero.functions.sum_to(gx0, self.x0_shape)
            gx1=dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
def div(x0, x1):
    x1=as_array(x1, cuda.get_array_module(x0.data))
    return Div()(x0,x1)
def rdiv(x0, x1):
    x1=as_array(x1, cuda.get_array_module(x0.data))
    return div(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c=c

    def forward(self, x):
        y=x**self.c
        return y

    def backward(self, gy):
        x,=self.inputs
        c=self.c

        gx=c*x**(c-1)*gy
        return gx
def pow(x,c):
    return Pow(c)(x)


def setup_variable():
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
    Variable.__getitem__=dezero.functions.get_item#[]접근
