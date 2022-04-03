import numpy as np

class Variable:
    def __init__(self, data):
        self.data=data
        self.grad=None#미분값을 저장할 for define-by-run
        self.creator=None#변수관점에서의 창조자 저장할 for 미분

    def set_creator(self, func):#변수와 함수의 관계를 y입장에서보면 function은 자신의 creator이다.
        self.creator=func#Define-by-Run 즉 실행시점에 계산을 연결하기 위해 사용된다

    def backward(self):#변수 관점으로 backwarding진행.
        f=self.creator
        if f is not None:
            x=f.input
            x.grad=f.backward(self.grad)
            x.backward()#하나 앞 변수의 backward호출(그 변수의 creator를 가져오고..반복 like 재귀)
            
class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y)
        
        output.set_creator(self)#output관점에서 현재의 function을 creator로서 설정한다.
        
        self.input=input
        
        self.output=output#현재의 output을 멤버함수에 저장해둔다.
        
        return output

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

#1. Define-by-Run의 이해
A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

assert y.creator==C#계산 그래프의 노드 거꾸로 올라가기. 변수의 creator, 함수의 input, input의 creator..
assert y.creator.input==b#이러한 형태의 connection이 계산실행시점에 만들어지는 것이 Define-by-Run이다.
assert y.creator.input.creator==B#LinkedList의 구조를 띈다.
assert y.creator.input.creator.input==a
assert y.creator.input.creator.input.creator==A
assert y.creator.input.creator.input.creator.input==x

#2. 역전파 상세과정
y.grad=np.array(1.0)

C=y.creator#y값의 creator
b=C.input#creator의 입력값
b.grad=C.backward(y.grad)#creator의 backward에 미분값 input, 그 결과를 creator의 입력값의 grad로 설정


B=b.creator#함수
a=B.input#입력
a.grad=B.backward(b.grad)#미분

A=a.creator
x=A.input
x.grad=A.backward(a.grad)
print(x.grad)

#3. Variable 클래스에 backward 메서드 추가
A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
c=C(b)

y.grad=np.array(1.0)
y.backward()
print(x.grad)
#y의 creator->creator의 input->input의 grad를 dy로->input의 창조자->창조자의 입력->입력의 grad를 창조자의 backward->...
