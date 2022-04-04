#의사코드. 가변길이를 역전파에도 적용하기 위해 Variable의 backward처리에 가변길이를 사용할 수 있도록 수정하였다.
#x와y들의 gradient를 의미하는 gxs와 gys로 zip을 통해 각각의 원소에 대한 x.grad를 gx로 갱신하는 둥의 일련의 과정을 취한다.
#여기서 주의해야할 점은 이전에 Function자체 구조 역시 가변인자를 지원하게 만들었기에 튜플을 반환한다는 점을 일단은 생각해야한다.
#이전에 원소개수하나면 그냥 반환하게 한거같긴한데..그건 call에 처리된거지 forward나 backward는 여전히 고려해야한다.
class Add(Function):
    def forward(self, x0, x1):
        y=x0+x1
        return y

    def backward(self, gy):#그냥 흘린다.
        return gy, gy

class Variable:
    ...
    def backward(self):
        if self.grad is None:
            self.grad=np.ones_like(self.data)

        funcs=[self.creator]
        while funcs:#여러 변수에 대응가능하게 수정
            f=funcs.pop()
            gys=[output.grad for output in f.outputs]#gradient of ys
            gxs=f.backward(*gys)#gradient of xs by backwarding

            if not isinstance(gxs, tuple):#만약 gxs가 tuple이 아니라면
                gxs=(gxs,)#튜플화

            for x, gx in zip(f.inputs, gxs):
                x.grad=gx#x변수에 대한 grad값으로 위에서 backwarding으로 구한 gx값을 대입

                if x.creator is not None:#변한거없고 다음 while시 pop을 위한 x의 creator append
                    funcs.append(x.creator)

class Square(Function):
    def forward(self, x):
        y=x**2
        return y

    def backward(self, gy):
        x=self.input[0].data#현재 Function클래스는 가변인자를 지원하기에
        gx=2*x*gy
        return gx
    
x=Variable(np.array(2.0))
y=Variable(np.array(3.0))

z=add(square(x), square(y))
z.backward()

print(z.data)#결과
print(x.grad)#각각의
print(y.grad)#미분값
