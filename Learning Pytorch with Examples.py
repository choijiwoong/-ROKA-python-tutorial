#[Simple Neural Network by numpy]
import numpy as np
import math

x=np.linspace(-math.pi, math.pi, 2000)
y=np.sin(x)

w=[np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()]

learning_rate=1e-6
for t in range(2000):
    y_pred=w[0]+w[1]*x+w[2]*x**2+w[3]*x**3#Get Predict_y=a+bx+cx^2+dx^3

    loss=np.square(y_pred-y).sum()#Cal loss
    if t%500==99:
        print(t, loss)

    grad_y_pred=2.0*(y_pred-y)#Cal gradient
    grad=[grad_y_pred.sum(), (grad_y_pred*x).sum(), (grad_y_pred*x**2).sum(), (grad_y_pred*x**3).sum()]#backpropagation
    for i in range(4):
        w[i]-=learning_rate*grad[i]#update weight
print(f'Result_numpy: y={w[0]}+{w[1]}x+{w[2]}x^2+{w[3]}x^3',end='\n\n')

#[Simple Neural Network by torch]
import torch

dtype=torch.float
device=torch.device("cpu")

x=torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y=torch.sin(x)

w=[torch.randn((), device=device, dtype=dtype), torch.randn((), device=device, dtype=dtype), torch.randn((), device=device, dtype=dtype), torch.randn((), device=device, dtype=dtype)]

learning_rate=1e-6
for t in range(2000):
    y_pred=w[0]+w[1]*x+w[2]*x**2+w[3]*x**3

    loss=(y_pred-y).pow(2).sum().item()
    if t%500==99:
        print(t,loss)

    grad_y_pred=2.0*(y_pred-y)
    grad=[grad_y_pred.sum(), (grad_y_pred*x).sum(), (grad_y_pred*x**2).sum(), (grad_y_pred*x**3).sum()]
    for i in range(4):
        w[i]-=learning_rate*grad[i]
print(f'Result_tensor: y={w[0].item()}+{w[1].item()}x+{w[2].item()}x^2+{w[3].item()}x^3',end='\n\n')

#[With Autograd]
dtype=torch.float
device=torch.device('cpu')

x=torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y=torch.sin(x)

w=[torch.randn((), device=device, dtype=dtype, requires_grad=True),#set
   torch.randn((), device=device, dtype=dtype, requires_grad=True),
   torch.randn((), device=device, dtype=dtype, requires_grad=True),#autograd를 설정해둔 tensor
   torch.randn((), device=device, dtype=dtype, requires_grad=True)]

learning_rate=1e-6
for t in range(2000):
    y_pred=w[0]+w[1]*x+w[2]*x**2+w[3]*x**3

    loss=(y_pred-y).pow(2).sum()#no use item()
    if t%500==99:
        print(t,loss)
    #알아서 calculate해준다. gradient를.
    loss.backward()#backpropagation. Now, element of w with .grad gets gradient. not it's tensor

    with torch.no_grad():
        for i in range(4):
            w[i]-=learning_rate*w[i].grad
            w[i].grad=None
print(f'Result_autograd: y={w[0].item()}+{w[1].item()}x+{w[2].item()}x^2+{w[3].item()}x^3',end='\n\n')

#[With definition new autograd function]_by subclass of torch.autograd.Function that defined forward, backward. In this example, use Legendre polynomial
class LegendrePolynomial3(torch.autograd.Function):#operation_사용자가 디자인한 함수를 포함하는 식을 forward하고 backward할 경우
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5*(5*input**3-3*input)
    @staticmethod
    def backward(ctx, grad_output):
        input,=ctx.saved_tensors
        return grad_output*1.5*(5*input**2-1)

dtype=torch.float
device=torch.device('cpu')

x=torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y=torch.sin(x)

w=[torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True),
   torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True),
   torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True),
   torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)]

learning_rate=5e-6
for t in range(2000):
    P3=LegendrePolynomial3.apply

    y_pred=w[0]+w[1]*P3(w[2]+w[3]*x)#P3 is subclass of torch.autograd.Function with forward & backward. so it can used by autograd

    loss=(y_pred-y).pow(2).sum()
    if t%500==99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        for i in range(4):
            w[i]-=learning_rate*w[i].grad
            w[i].grad=None
print(f'Result_LegendrePolynomial3: y={w[0].item()}+{w[1].item()}x+{w[2].item()}x^2+{w[3].item()}x^3',end='\n\n')

#[With nn module]_like Keras, TensorFlow-Slim, FLearn in Tensorflow(abstraction)
x=torch.linspace(-math.pi, math.pi, 2000)#2000 tensor [-math.pi, math.pi]
y=torch.sin(x)

p=torch.tensor([1,2,3])
xx=x.unsqueeze(-1).pow(p)#shape(2000,3)

model=torch.nn.Sequential(#sqeuence of layer
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)#start dim, end dim. maybe xx is 2d_match shape of y
)#여러 좋은 툴들로 보다 좋은 추상화를 할 수 있게끔

loss_fn=torch.nn.MSELoss(reduction='sum')#Mean Squared Error. sum: 출력이 합산, mean: 출력합이 요소수로 나뉨

learning_rate=1e-6
for t in range(2000):
    y_pred=model(xx)#make tensor of output data

    loss=loss_fn(y_pred, y)
    if t %500==99:
        print(t, loss.item())
    #마찬가지로 편리한 툴들은 서비스 
    model.zero_grad()#make gradient to Zero befor backpropagation
    loss.backward()#backpropagation_calculate gradient

    with torch.no_grad():#update weights(parameters) by GCD_w=w-a*f'
        for param in model.parameters():
            param-=learning_rate*param.grad#grad is calculated by loss.backward.
linear_layer=model[0]#access to first layer
print(f'Result_nn: y={linear_layer.bias.item()}+{linear_layer.weight[:,0].item()}x+{linear_layer.weight[:,1].item()}x^2+{linear_layer.weight[:,2].item()}x^3', end='\n\n')

#[With optimizer]_SGD, AdaGrad, RMSProp, Adam
x=torch.linspace(-math.pi, math.pi, 2000)
y=torch.sin(x)

p=torch.tensor([1,2,3])
xx=x.unsqueeze(-1).pow(p)

model=torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)
loss_fn=torch.nn.MSELoss(reduction='sum')

learning_rate=1e-3
optimizer=torch.optim.RMSprop(model.parameters(), lr=learning_rate)
for t in range(2000):
    y_pred=model(xx)

    loss=loss_fn(y_pred,y)
    if t%500==99:
        print(t, loss.item())
    #autograd로 자동 gradient계산까지 했다면, optimzer에 model parameter들을 전달하여 바로 update까지 할 수 있게 도구제공!
    optimizer.zero_grad()#set weight's grad to zeroby optimizer for preventing accumulation

    loss.backward()#calculate gradient to all model parameter to .grad

    optimizer.step()#update weight by optimizer; RMSprop
linear_layer=model[0]
print(f'Result_optimizer(RMPprop): ={linear_layer.bias.item()}+{linear_layer.weight[:,0].item()}x+{linear_layer.weight[:,1].item()}x^2+{linear_layer.weight[:,2].item()}x^3', end='\n\n')


#[With user defined nn.Module]
class Polynomial3(torch.nn.Module):#모델 자체를 사용자가 직접 디자인하고 싶을 때. 
    def __init__(self):
        super().__init__()

        self.a=torch.nn.Parameter(torch.randn(()))
        self.b=torch.nn.Parameter(torch.randn(()))
        self.c=torch.nn.Parameter(torch.randn(()))
        self.d=torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a+self.b*x+self.c*x**2+self.d*x**3

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

x=torch.linspace(-math.pi, math.pi, 2000)
y=torch.sin(x)

model=Polynomial3()#인스턴스화하여 사용
print('model parameters: ', model.parameters())
criterion=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(2000):
    y_pred=model(x)#forward

    loss=criterion(y_pred, y)
    if t%500==99:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'Result_user defined nn.Module_Polynomial3: {model.string()}', end='\n\n')


#[With Control flow & Weight sharing]
import random

class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.a=torch.nn.Parameter(torch.randn(()))#make it as parameter
        self.b=torch.nn.Parameter(torch.randn(()))
        self.c=torch.nn.Parameter(torch.randn(()))
        self.d=torch.nn.Parameter(torch.randn(()))
        self.e=torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        y=self.a+self.b*x+self.c*x**2+self.d*x**3#common expression
        for exp in range(4, random.randint(4,6)):#대충 뭘 보라고 만든 예시인지 잘 이해안가지만 forward는 dynamic이기에 일반적인 연산자들이 가능하다는 것 같고
            #가중치 공유는 찾아보니 CNN에서 하나의 커널이 뉴런의 볼륨을 stride하여 모든 커널이 동이한 가중치를 갖는다는 것을 의미한다는 건데..
            #단순히 재사용되어 같은 값을 가진다는 의미에서 가중치 재사용이라고했나..그럼 이 예제의 목적은 뭐지...그냥 동일 매개변수를 여러번 재사용할수있다는 거네 그 매개변수가 곧 weight니까
            y=y+self.e*x**exp#add ex^exp(random) to expression
        return y

    def string(self):
        return f'y={self.a.item()}+{self.b.item()}x+{self.c.item()}x^2+{self.d.item()}x^3+{self.e.item()}x^4 ? +{self.e.item()}x^5 ?'

x=torch.linspace(-math.pi, math.pi, 2000)
y=torch.sin(x)

model=DynamicNet()#instantiation

criterion=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)#momentum!

for t in range(30000):
    y_pred=model(x)#get prediction
    
    loss=criterion(y_pred, y)#cal los
    if t%5000==1999:
        print(t, loss.item())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f'Result_for understanding of Control Flow & Weight Sharing: {model.string()}')
