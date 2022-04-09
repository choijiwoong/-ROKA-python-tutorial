#간단한 오버로딩 by 특수 메서드

#의사코드
class Mul(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y

    def backward(self, gy):
        x0, x1=self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0#거꾸로
def mul(x0, x1):
    return Mul()(x0, x1)

a=Variable(np.array(3.0))
b=Variable(np.array(2.0))
c=Variable(np.array(1.0))

y=add(mul(a,b), c)
y.backward

print(y)
print(a.grad)
print(b.grad)

#연산자 오버로드
Variable:
    ...
    def __mul__(self, other):
        return mul(self other)
Variable.__add__=add#처럼 해당 클래스의 특수 메서드에 직접 우리가 위에서 만든 함수를 할당할 수도 있다.

a=Variable(np.array(3.0))
b=Variable(np.array(2.0))
y=a*b#__mul__호출. a는 self로 b는 other로
print(y)

a=Variable(np.array(3.0))
b=Variable(np.array(2.0))
c=Variable(np.array(1.0))

#y=add(mul(a,b),c) 기존의 방식
y=a*b+c
y.backward

print(y)
print(a.grad)
print(b.grad)
