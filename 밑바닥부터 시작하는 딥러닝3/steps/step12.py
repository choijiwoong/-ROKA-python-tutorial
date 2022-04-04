from step11 import *


class Function:
    def __call__(self, *inputs):#1) 함수를 사용하기 쉽게 튜플이나 리스트를 거치지 않고 가변인수와 결과를 직접 주고받는다.
        xs=[x.data for x in inputs]
        ys=self.forward(*xs)#list unpack
        if not isinstance(ys, tuple):#tuple입력이 아닌경우 처리
            ys=(ys,)
        outputs=[Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=outputs

        return outputs if len(outputs)>1 else outputs[0]#출력이 하나면 그냥 원소를 반환

def f(*x):
    print(x)
f(1,2,3)
f(1,2,3,4,5,6)

class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y
def add(x0, x1):
    return Add()(x0, x1)

x0=Variable(np.array(2))
x1=Variable(np.array(3))
y=add(x0, x1)
print(y.data)
