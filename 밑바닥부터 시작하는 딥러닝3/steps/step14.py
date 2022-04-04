#의사코드

#현재의 문제1: 같은 변수를 반복하여 사용할 경우 의도대로 동작하지 않을 수 있다
x=Variable(np.array(3.0))
y=add(x,x)#같은 변수 사용
print('y', y.data)#6

y.backward()
print('x.grad', x.grad)#1 출력. y=2*x의 미분은 원래 2이다. 아마 단순히 흘리는 것에 초점을 두어 여러개가와도 그대로 흘린게 문제일 것으로 추측한다.

#문제의 원인
class Variable:
    ...
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:
            f=funcs.pop()
            gyx=[output.grad for output in f.outputs]
            gxs=f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs=(gxs,)

            for x, gx in zip(f.inputs, gxs):
                #x.grad=gx#전해지는 미분값을 대입하는데, 이때 같은 변수를 사용하는 경우 그냥 덮어씌워지기 때문이다.
                #해결책
                if x.grad is None:#null일때만 대입을 하고
                    x.grad=gx
                else:#이미 있는 경우엔 가산한다.(no cover)
                    x.grad=x.grad+gx
                
                if x.creator is not None:
                    funcs.append(x.creator)
                

x=Variable(np.array(3.0))
y=add(x,x)
y.backward()
print(x.grad)#올바른 2.0의 결과를 얻는다.

x=Variable(np.array(3.0))
y=add(add(x,x),x)
y.backward()
print(x.grad)#3번 겹쳐도 3.0의 올바른 결과를 얻는다. x.grad세팅시 덮어쓰기가 아닌 누산을 했기 때문이다.


#위의 해결책이 불러온 또다른 문제. 메모리 재활용 시 미분값 초기화의 필요성
x=Variable(np.array(3.0))
y=add(x,x)
y.backward()
print(x.grad)#2.0의 정상적인값

y=add(add(x,x),x)#메모리를 아끼기 위해 기존의 변수 이용
y.backward()
print(x.grad)#5.0(3.0이 이상적) 즉, 이전의 미분값 grad가 2.0인 상태로 초기화되어있지 않은 채로 정답인 3.0이 누산되어 5.0이 되었다.

class Variable:
    ...
    def cleargrad(self):#을 추가하여 이전의 미분값을 초기화시킬 수 있다.
        self.grad=None

x=Variable(np.array(3.0))
y=add(x,x)
y.backward()
print(x.grad)#2.0

x.cleargrad()#x의 재활용을 위한 grad값 None화
y=add(add(x,x),x)
y.backward()
print(x.grad)#3.0
