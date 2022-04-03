"""
class Function:#Variable입력을 받아 제곱해준다.
    def __call__(self, input):
        x=input.data
        y=x**2
        output=Variable(y)
        return output

x=Variable(np.array(10))
f=Function()
y=f(x)

print(type(y), y.data)#<class '__main__.Variable'> 100"""
from step01 import *

#앞으로의 유용한함수들의 사전 구현에 Function을 기반클래스로 사용한다.
class Function:#Variable입력을 받아 제곱해준다.
    def __call__(self, input):#호출시에는 input된 Variable클래스의 데이터를 가져오고
        x=input.data
        y=self.forward(x)#구체적인 연산은 forward에서 실행한다.
        output=Variable(y)#결과를 Variable로 변환 후 리턴
        return output

    def forward(self, x):#기반클래스 자체에서 forward를 제공하지 않는다.
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x**2
"""
x=Variable(np.array(10))
f=Square()#인스턴스화
y=f(x)
print(type(y), y.data)"""
