#중간 총 정리_격리하며 andrew ng, 경영 강의 다 듣고 밑바닥 딥러닝 3권 마무리하는거 목표잡자ㅋㅋㅋㅋㅋ원래 이번주 목표 다할수있겠노! 그 이상으로 늘어나는 경우를 대비해서 os책 하나 들고가고ㅎㅎ 아니다 만약 이번주 목표  다 끝나면 남은 격리는 나를 위한 시간으로..
"""복잡한 구현의 경우 현재 Variable은 제 역활을 못하는데, repeat노드를 사용하는 경우
미분시 양쪽의 결과를 sum한 다음 그 결과를 전파해야하는데 funcs라는 creator(function)리스트 append와 pop을 사용한 특성 상
한쪽의 결과가 끝까지 역전파 된 후 나머지 repeat노드로의 역전파가 더온다. 즉 올바르지 않은 역전파가 여러번 일어난다.
 이러한 문제를 해결하기 위해 함수에 우선순위를 주기 위해 창조자-피조물 관계 혹은 부모-자식 관계의 계층별로 진행하는 Topological Sort알고리즘을 사용한다.
고로 Variable클래스와 Function클래스에 인스턴스 변수 generation을 추가한다."""
import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:#ndarray만 받는다
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은 지원하지 않습니다.'.format(type(data)))

        self.data=data
        self.grad=None
        self.creator=None#Define-By-Run
        self.generation=0#repeat와 같은 경우에 기존 append pop방식의 단점을 보완하기 위한 우선순위 기준

    def set_creator(self, func):
        self.creator=func
        self.generation=func.generation+1#Variable의 generation을 creator.generation+1값으로

    def cleargrad(self):#for 재활용
        self.grad=None

    def backward(self):
        if self.grad is None:#조건부 초기화
            self.grad=np.ones_like(self.data)

        funcs=[]
        seen_set=set()#중복추가 실수방지
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f=funcs.pop()#함수를 generation높은거부터 빼게 됨.(이미 정렬되어있어서)
            gys=[output.grad for output in f.outputs]#dout
            gxs=f.backward(*gys)#dx
            if not isinstance(gxs, tuple):
                gxs=(gxs,)#to tuple

            for x, gx in zip(f.inputs, gxs):#x, dx
                if x.grad is None:
                    x.grad=gx#cover
                else:#repeat노드 대비
                    x.grad=x.grad+gx

                if x.creator is not None:#추가 creator가 있다면
                    add_func(x.creator)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):#get variadic arguments
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)
        if not isinstance(ys, tuple):#y to tuple
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]#y to Variable

        self.generation=max([x.generation for x in inputs])#현재 함수의 세대는 입력값들 세대중 최고값으로.
        for output in outputs:
            output.set_creator(self)#내부적으로 generation을 현재 함수의 generation보다 +1로 세팅
        self.inputs=inputs
        self.outputs=outputs
        return outputs if len(outputs)>1 else outputs[0]#scalar면 원소를 반환(함수)

    def forward(self, xs):#abstract
        raise NotImplementedError()

    def backward(self, gys):#abstract
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y=x**2
        return y

    def backward(self, gy):
        x=self.inputs[0].data#어차피 단일 원소일테니.
        gx=2*x*gy
        return gx
def square(x):
    return Square()(x)

class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y

    def backward(self, dy):
        return dy, dy#흘리기
def add(x0, x1):
    return Add()(x0, x1)

"""test: sort by generation
generations=[2,0,1,4,2]
funcs=[]

for g in generations:#임의의 세대정보로 Function만들고 append
    f=Function()
    f.generation=g
    funcs.append(f)
print([f.generation for f in funcs])

funcs.sort(key=lambda x: x.generation)#lambda이용, generation기준 오름차순 정렬(heap이용가능)
print([f.generation for f in funcs])
print(funcs.pop().generation)#pop시 최상 generation출력

#test2
x=Variable(np.array(2.0))
a=square(x)
y=add(square(a), square(a))#이전에 풀지 못했던 repeat노드에 대한 test after priority by generation
y.backward()
print(y.data)#32.0
print(x.grad)#64.0 well done!
"""
