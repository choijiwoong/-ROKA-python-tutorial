import numpy as np#가변길이 인수
from step09 import *

class Function:
    def __call__(self, inputs):
        xs=[x.data for x in inputs]#리스트형태를 받음
        ys=self.forward(xs)
        outputs=[Variable(as_array(y)) for y in ys]#함수 call값을 array형태의 Variable 리스트로

        for output in outputs:
            output.set_creator(self)
        self.inputs=inputs
        self.outputs=outputs
        return outputs

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, xs):
        x0, x1=xs#리스트형태의 값 2개를 분해하여
        y=x0+x1#합한결과를
        return (y,)#리스트형태로 리턴
"""
xs=[Variable(np.array(2)), Variable(np.array(3))]
f=Add()
ys=f(xs)
y=ys[0]
print(y.data)
아쉬운 점은 사용자가 입력 변수를 리스트에 담아 건네주며, 반환값은 튜플로 받는다."""
