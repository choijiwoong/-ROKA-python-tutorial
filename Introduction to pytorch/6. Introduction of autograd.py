"""
requires_grad속성을 True로 해두는 순간 backward를 위한 trace가 시작되며, grad_fn.next_functions를 이용하여 drill down하여 None인 gradient까지 접근이 가능하다.
항상 optimizer.zero_grad()호출 후 optimizer.step()을 호출해야지 그렇지 않으면 loss.backward()의 학습 가중치가 update에 사용되지 못하고 누적된다.
autograd사용 시 내부 작업을 진행하며 autograd를 계산하는데 필요한 정보가 손실될 가능성이 있는 것에 유의해야한다.
autograd는 계산의 모든단계를 자세히 추적하는데, 이러한 계산 기록으로 프로파일러를 만들 수 있으며 이는 autograd에 내장되어있다.
 n차원 입력과 m차원 출력의 완전한 기울기는 Jacobian미분행렬로 표현되는데, 구체적으로 PyTorch모델, 손실함수로 예시를 들면, PyTorch모델에 loss기울기를 곱하고 연쇄규칙을 적용하면
열벡터, 첫번째 입력에 대한 두번째 함수의 기울기를 얻을 수 있고 이는 손실 기울기이다. torch.autograd는 이러한 작업을 위한 엔진으로 backpropagation간 learning weight의 gradient를 accumulate한다.
 고수준 API로 중요한 미분 행렬, 벡터 연산에 직접 접근이 가능하게 하는 API가 autograd에 내장되어 있어 특정 입력에 대한 특정 함수의 야코비안 행렬 및 헤세 행렬을 계산할 수 있다.
(헤세 함수는 야코비안 행렬과 비슷하나, 모든 2차 도함수를 표현한다. 이는 Convex Optimization, 이계도함수 판정, Image processing(vessel detection)등에 사용된다.
기본적인 아이디어는 Hessian행렬이 함수의 Bowl형태가 얼마나 변형되었는가를 나타내주는 것이다. Hessian Matrix의 eigenvector들은 변형의 prinipal axis를 나타내며
eigenvalue들은 변형정도를 나타내게 된다. specific point에서의 Hessian eigenvalue차이가 크가면 point는 길쭉한 모양인 것이다.)
또한 이러한 행렬로 벡터 곱을 구하는 방법을 제공한다.
 torch.autograd.functional.hessian()는 torch.autograd.functional.jacobian과 동일하게 작동하지만 모든 2차 도함수의 행렬을 반환하며, 벡터를 제공하는 경우 벡터-야코비안 곱을 직접 계산할 수 있는데,
피연산자가 반전된 경우들을 포함하여 vjp(), vhp(), hvp()함수를 지원한다.
"""
#[Simple example of autograd]
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

a=torch.linspace(0., 2. *math.pi, steps=25, requires_grad=True)#make input tensor as same stride value
print(a)#thanks to requires_grad=True, all sequenced calculation will be recorded to output tensor


b=torch.sin(a)
plt.plot(a.detach(), b.detach())
#plt.show()

print(b)#show indicator that is tracing calculations; grad_fn=<SinBackward0>
#more calculate...
c=2*b
print(c)#grad_fn=<MulBackward0>

d=c+1
print(d)#grad_fn=<AddBackward0>

out=d.sum()
print(out)#grad_fn=<SumBackward0>

#Let's drill down!
print('\n\nd: ')
print(d.grad_fn)
print(d.grad_fn.next_functions)#we can approach to gradient until None
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)#<AccumulateGrad>
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)#()
print('\nc: ')
print(c.grad_fn)#<MulBackward0>
print('\nb: ')
print(b.grad_fn)#<SinBackward0>
print('\na: ')
print(a.grad_fn)#None

#a=torch.linspace(0., 2.*math.pi, steps=25, requires_grad=True)
#b=torch.sin(a)
#c=2*b
#d=c+1
#out=d.sum()

out.backward()
print(a.grad,end='\n\n[Autograd in learning]\n')
plt.plot(a.detach(), a.grad.detach())
#plt.show()


#[Autograd on/off]
#modify autograd
a=torch.ones(2,3,requires_grad=True)#후.. 아까의 simple learning교훈을 따와 print잘 구분해두어야겠다..
print("Tensor with autograd: ", a)

b1=2*a
print("After Calculation with autograd: ", b1)

a.requires_grad=False
b2=2*a
print("After Calculation without autograd: ",b2, end='\n\n')#No grad_fn because tracing is non-activated thanks to requires_grad.

#example of turnning off autograd
a=torch.ones(2,3,requires_grad=True)*2
b=torch.ones(2,3,requires_grad=True)*3
c1=a+b
print("a+b with autograd: ", c1)

with torch.no_grad():
    c2=a+b
print("a+b without autograd by with-resource: ", c2)

c3=a*b
print("a*b with autograd: ", c3,end='\n\n')

#Another way for turnning off autograd by @torch.no_grad()
def add_tensors1(x,y):
    return x+y

@torch.no_grad()
def add_tensors2(x,y):
    return x+y

a=torch.ones(2,3, requires_grad=True)*2
b=torch.ones(2,3, requires_grad=True)*3

c1=add_tensors1(a,b)
print("use add_tensors1 method that's normal function:", c1)
c2=add_tensors2(a,b)
print("use add_tensor2 method that has @torch.no_grad():", c2, end='\n\n')#thanks to #torch.no_grad()

#The other way for turnning off autograd by detach()_copy to record of calculation
x=torch.rand(5, requires_grad=True)
y=x.detach()#detach to x!
print("Normal tensor with autograd:", x)
print("detached tensor by x: ",y, end='\n\n')#no autograd!


#[Autograd in place]
print("[Autograd in place]")

#problem_inner modifying with autograd
try:
    a=torch.linspace(0., 2.*math.pi, steps=25, requires_grad=True)#with autograd
    torch.sin_(a)#work in place
except RuntimeError as e:
    print(e, end='\n\n')

#solution: use profiler
device=torch.device('cpu')
run_on_gpu=False

if torch.cuda.is_available():
    device=torch.device('cuda')
    run_on_gpu=True
    
x=torch.randn(2,3, requires_grad=True)
y=torch.rand(2,3, requires_grad=True)
z=torch.ones(2,3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:#with profiler
    for _ in range(1000):
        z=(z/x)*y#do it 1000times
print(prf.key_averages().table(sort_by='self_cpu_time_total'), end='\n\n')#get information by profiler


#[Details of Autograd & Advanced API] 부록
#details of autograd
x=torch.randn(3, requires_grad=True)
y=x*2
while y.data.norm()<1000:
    y=y*2#sequencely multiple 2
print("y.data.norm()>=1000 now: ",y)

try:
    y.backward()
except RuntimeError as e:
    print("error when y.backward: ",e)#지금은 스칼라 출력에 대해서만 backward가능

v=torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print("x.grad that is multi-dimention tensor: ", x.grad, end='\n\n')

#advanced api
def exp_adder(x, y):
    return 2*x.exp()+3*y#2*e^x+3*y

inputs=(torch.rand(1), torch.rand(1))#simple jacobian matrix
print("inputs: ", inputs,end='\n\n')
print("after exp_adder: ",torch.autograd.functional.jacobian(exp_adder, inputs))

inputs=(torch.rand(3), torch.rand(3))
print(inputs)
print("after exp_adder: ", torch.autograd.functional.jacobian(exp_adder, inputs), end='\n\n')

#function to directly compute the vector-jacobian product
def do_some_doubling(x):
    y=x*2
    while y.data.norm()<1000:
        y=y*2
    return y
inputs=torch.randn(3)
my_gradients=torch.tensor([0.1, 1.0, 0.0001])
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)#vector-jacobian product
