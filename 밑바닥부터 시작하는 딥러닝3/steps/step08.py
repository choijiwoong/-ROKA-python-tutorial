#Variable의 backward를 재귀꼴로 변환. 기존의 방식은 변수든 creator든 모두 재귀를 타고갔다
class Variable:#의사코드
    ...
    def backward(self):
        funcs=[self.creator]#현재 변수의 함수를 가져와
        while func:#존재한다면
            f=funcs.pop()#꺼내서
            x,y=f.input, f.output#입출력을 가져와서
            x.grad=f.backward(y.grad)#backward호출

            if x.creator is not None:#만약 x의 creator가 있으면 append하여 연쇄적으로 사용이 가능하게 하고, 그게 아니면 while조건어겨 escape
                funcs.append(x.creator)

A=Square()
B=Exp()
C=Square()

x=Variable(np.array(0.5))
a=A(x)
b=B(a)
y=C(b)

y.grad=np.array(1.0)
y.backward()#y에 대해 backwarding을 하면
print(x.grad)#x.grad에 최종적인 값이 세팅된다.
