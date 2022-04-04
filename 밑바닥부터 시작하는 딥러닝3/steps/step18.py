#의사코드

#1. 필요없는 중간 미분값 인자 retain_grad에 따라 삭제
x0=Variable(np.array(1.0))
x1=Variable(np.array(1.0))
t=add(x0, x1)
y=add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)#즉, t와 중간 변수들까지 grad값을 기록하는데 역전파로 보통 구하면 말단 변수만 필요하다. 고로 인자로 정보를 받아 필요없는 미분값을 삭제한다.
class Variable:
    ...
    def backward(self, retain_grad=False):#grad가 필요하지 않다.
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_set=set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_funcs(self.creator)

        while funcs:
            f=funcs.pop()
            gys=[output().grad for output in f.outputs]
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

            if not retain_grad:#필요하지 않다면
                for y in f.outputs:#결과값의 grad정보를 삭제한다.
                    y().grad=None#y 즉 outputs은 저번에 순환참조 없앤다고 weakref로 했기에 접근을 위해선 ()로.

x0=Variable(np.array(1.0))
x1=Variable(np.array(1.0))
t=add(x0,x1)
y=add(x0,t)
y.backward()

print(y.grad, t.grad)#중간값의 grad는 None
print(x0.grad, x1.grad)#말단은 잘 나옴


#2. Function의 input값의 경우 backward를 위한 cache인데 forward만 진행하는 연산일 경우 저장할 필요가 없고 낭비가 된다.
#고로 forward모드와 forward&backward를 모드를 상황에 따라 전달해주는 방법이 효율적이다.

 #1) 첫번째 방법으로 별도의 클래스를 사용할 수 잇다.
class Config:
    enable_backprop=True

class Function:
    def __call__(self, *inputs):
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys, tuple):
            ys=(ys,)
        outputs=[variable(as_variable(y)) for y in ys]

        if Config.enable_backprop:#Config의 변수가 True일 때만 backward를 위한 처리를 진행한다.
            self.generation=max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs=inputs
            self.outputs=[weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs)>1 else outputs[0]

Config.enable_backprop=True#backward를 가정하고 cache계산 및 저장
x=Variable(np.ones((100,100,100)))
y=square(square(square(x)))
y.backward()

Config.enable_backprop=False#forward만을 가정하고 불필요한 cache및 연산 X
x=Variable(np.ones((100,100,100)))
y=square(square(square(x)))

 #2) 두번째 방법으로 with문을 사용하여 모드를 전환시킬 수 있다.
with using_config('enable_backprop', False):#이 영역 안에서만 역전파 비활성. 벗어나면 다시 활성영역
    x=Variable(np.array(2.0))
    y=square(x)

#위와 같은 with문법은 아래와 같이 사용이 가능하다.
import contextlib

@contextlib.contextmanager
def config_test():
    print('start')#전처리
    try:
        yield
    finally:
        print('done')#후처리

with config_test():
    pritn('process...')

#실제 구현해보자
class Config:
    enable_backprop=True

@contextlib.contextmanager
def using_config(name, value):
    old_value=getattr(Config, name)#Config 클래스에 대한 속성의 이름을 가져온다.
    setattr(Config, name, value)#Config에서 위에서 구한 속성의 값을 인자인 value로 설정하고
    try:
        yield
    finally:#모든 처리가 끝난뒤에
        setattr(Config, name, old_value)#백업해둔 이전의 상태로 돌려둔다.

with using_config('enable_backprop', False):#내부적으로 지칭하는 Config의 속성 enable_backprop을 False로 설정하고 아래의 작업을 진행한 뒤
    x=Variable(np.array(2.0))
    y=square(x)#scope탈출시 돌려둔다.
    
#편의를 위해 Value를 직접 설정하는 함수를 제작
def no_grad:
    return using_config('enable_backprop', False)

with no_grad():
    x=Variable(np.array(2.0))
    y=square()
