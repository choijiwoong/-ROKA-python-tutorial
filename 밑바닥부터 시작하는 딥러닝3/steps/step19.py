#중간총정리_메모리파트
#Variable에 이름을 지정시키는 기능, shape ndim같은 유용한 인스턴스 변수, len과 print함수기능을 추가했다.

import weakref#순환참조 개선용
import numpy as np
import contextlib#with를 통한 no_grad모드 전환을 위함

class Config:#모드의 설정을 위한 클래스
    enable_backprop=True

@contextlib.contextmanager
def using_config(name, value):#속성의 이름과 값을 입력받고
    old_value=getattr(Config, name)#Config클래스의 현재값을 백업한뒤
    setattr(Config, name, value)#argument value로 설정한다.
    try:
        yield
    finally:
        setattr(Config, name, old_value)#scope탈출 시 원상복구

def no_grad():#편의를 위한 함수. 모듈 전체의 기본설정이 yes_grad라 no_grad만 모드전환용으로 구현
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data=data
        self.name=name
        self.grad=None
        self.creator=None
        self.generation=0
        
    #유용한 인스턴스 변수들
    @property
    def shape(self):#인스턴스.shape
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
    
    #유용한 특수메서드
    def __len__(self):
        return len(self.data)

    def __repr__(self):#출력시
        if self.data is None:
            return 'variable(None)'
        p=str(self.data).replace('\n', '\n'+' '*9)#?
        return 'variable('+p+')'

    #일반 멤버함수
    def set_creator(self, func):
        self.creator=func
        self.generation=func.generation+1#출력은 함수+1세대

    def cleargrad(self):#재활용
        self.grad=None

    def backward(self, retain_grad=False):#쓸데없는 중간 미분값 필요없어욧!
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        func=[]
        seen_set=set()

        def add_func(f):
            if f not in seen_set:#중복미분실수방지
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]#output은 순환참조의 문제를 깨기위해 weakref로 설정해둠. 값접근을 위해 ()로. creator는 그대로
            gxs=f.backward(*gys)#unpack
            if not isinstance(gxs, tuple):#dx결과 tuple화
                gxs=(gxs,)

            for x, gx in zip(f.inputs, gxs):#구한 dx를 x.grad에 update
                if x.grad is None:
                    x.grad=gx
                else:#repeat노드처럼 덮어쓸 경우를 위한 누산
                    x.grad=x.grad+gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:#모든 기울기가 필요없다고 했으면(유저가)
                for y in f.outputs:#x를 제외한 y에 대하여
                    y().grad=None#위에 말했다시피 output은 weakref임

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]#x들
        ys=self.forward(*xs)#y들
        if not isinstance(ys, tuple):#tuple화
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]#Variable화

        if Config.enable_backprop:#미분이 목적이라면
            self.generation=max([x.generation for x in inputs])#세대 잘 설정하고
            for output in outputs:
                output.set_creator(self)#creator잘 설정하고
            self.inputs=inputs#cache
            self.outputs=[weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y=x**2
        return y

    def backward(self, gy):
        x=self.inputs[0].data
        gx=2*x*gy
        return gx
def square(x):
    return Square()(x)

class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y

    def backward(self, gy):
        return gy, gy
def add(x0, x1):
    return Add()(x0, x1)

x=Variable(np.array([[1,2,3], [4,5,6]]))
x.name='x'

print(x.name)
print(x.shape)
print(x)
