"""
1차원 tensor는 벡터라고 부르기도 하며, 2차원 tensor도 행렬이라 부르기도하며, 나머지는 tensor로 통칭한다.
 랜덤값의 재현성 관점에서, CUDA 컴볼루션 작업에서 사용하는 cuDNN은 비결정성의 소스가 될 수 있기에 torch.backends.cudnn.benchmark=False로 비활성화 할 수 있지만
성능저하의 가능성이 있기에 torch.use_deterministic_algorithms(True)사용가능한경우 비결정적 알고리즘 대신 결정적 알고리즘을 사용하게 할 수 있다.
 대부분의 tensor연산은 같은 모양텐서끼리만 가능하지만, 예외로 비슷한 모양의 텐서간의 연산을 수행하는 텐서 브로드캐스팅이 있다. 일반적인 사용 예로는
학습가중치텐서에 입력텐서배치를 곱하고 배치의 각 인스턴스에 개별적 연산적용 후 동일한 모양의 텐서를 반환하는 것이다. 브로드캐스팅의 조건으로는
최소 하나의 차원을 가지며 빈텐서는 없어야한다. 또한 각 차원은 동일하거나 치수중 하나는 크기가 1이어야 하거나 차원이 없어야한다.
 torch.cross는 벡터의 외적을 의미하는데, a cross b시 방향은 a와b에 동시수직, 크기는 a와b를 변으로하는 평행사변형 넓이이다. a=(a1,a2,a3), b=(b1,b2,b3)일때
외적은 aXb=(a2b3-a3b2, a3b1-a1b3, a1b2-a2b1)이다.
 clone()이용한 deepcopy 시 기본 option setting또한 복사되기에 소스 텐서에 autograd가 있으면 클론도 활성화된다. 일반적인 경우엔 오히려좋아~로 사용하겠지만
모든것에 대한 gradient가 기본적으로 off이지만, 일부 메트릭 생성을 위해 일부 값을 중간에 꺼내려는 경우 모델이 계산수행중인 경우에는
복제된 복사본이 gradient를 추적하는것을 원하지 않아하기에 .detach()로 autograd의 히스토리 추적을 끄면 된다.
 GPU역시 전용 메모리가 연결되어있어, CUDA에서 계산을 수행하려면 GPU가 엑세스할 수 있는 메모리로 데이터를 이동시켜야한다.
 보통의 경우 squeeze, unsqueeze로 웬만한 차원의 조작이 가능하지만, 원소들의 개수, 내용을 유지하며 텐서의 모양을 근본적으로 변경시키고싶다면 reshape()로 수행이 가능한데
이는 보통 모델의 컨볼루션 레이어와 모델의 선형 레이어 사이의 인터페이스로 사용된다.
 pytorch의 tensor shape는 튜플을 받기에 웬만하면 정수보다 (dimention,)으로 한요소 튜플임을 명시하자.
"""
import torch
import math

#make tensor
x=torch.empty(3,4)
print(type(x))
print(x, end='\n\n')

#factory method of tensor
zeros=torch.zeros(2,3)
print(zeros)

ones=torch.ones(2,3)
print(ones)

torch.manual_seed(1729)
random=torch.rand(2,3)#float 0~1
print(random, end='\n\n')

#seed of random tensor
torch.manual_seed(1729)
random1=torch.rand(2,3)
print(random1)
random2=torch.rand(2,3)
print(random2)

torch.manual_seed(1729)
random3=torch.rand(2,3)
print(random3)#same value with random1 because of seed
random4=torch.rand(2,3)
print(random4, end='\n\n')#same value with random2

#shape of tensor
x=torch.empty(2,2,3)
print(x.shape)
print(x)

empty_like_x=torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x=torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x=torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x=torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x, end='\n\n')

#make tensor by data directly
some_constants=torch.tensor([[3.1414926, 2.71828], [1.61803, 0.0072897]])#make multidimensional tensor by nested collections
print(some_constants)
some_integers=torch.tensor((2,3,5,7,11,13,17,19))
print(some_integers)
more_integers=torch.tensor(((2,4,6),[3,6,9]))
print(more_integers, end='\n\n')

#data type of tensor; torch.bool, int8, uing8, int16, int32, int64, half(float와 비슷), float, double, bfloat
a=torch.ones((2,3), dtype=torch.int16)
print(a)
b=torch.rand((2,3), dtype=torch.float64)*20.
print(b)
c=b.to(torch.int32)#casting to tensor by to method
print(c, end='\n\n')

#logic & math on tensor
ones=torch.zeros(2,2)+1
twos=torch.ones(2,2)*2
threes=(torch.ones(2,2)*7-1)/2
fours=twos**2
sqrt2s=twos**0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

powers2=twos**torch.tensor([[1,2],[3,4]])
print(powers2)
fives=ones+fours
print(fives)
dozens=threes*fours
print(dozens)

try:#upper calculations are on same shape tensor. if shapes are different, throw runtime error
    a=torch.rand(2,3)
    b=torch.rand(3,2)
    print(a*b)
except RuntimeError as e:
    print(e, end='\n\n')

#exception of 'calculation on same shape rule': tensor broadcasting***
rand=torch.rand(2,4)
doubled=rand*(torch.ones(1,4)*2)#dimention 1! satisfy condition of broadcasting
print(rand)
print(doubled)

a=torch.ones(4,3,2)
b=a*torch.rand(3,2)#3rd&2nd dims identical to a, dim 1 absent
print(b)
c=a*torch.rand(3,1)#3rd dim=1, 2nd dim identical to a
print(c)
d=a*torch.rand(1,2)#3rd dim identical to a, 2nd dim=1
print(d)

#fail of broadcasting(wrong example)
try:
    a=torch.ones(4,3,2)
    b=a*torch.rand(4,3)#(0,4,3)
    c=a*torch.rand(2,3)#(0,2,3)
    a=a*torch.rand((0,))#empty tensor
except RuntimeError as e:
    print(e,end='\n\n')

#more mathmetic calculation by tensor
a=torch.rand(2,4)*2-1
print('Common functions: ')
print(torch.abs(a))
print(torch.ceil(a))
print(torch.floor(a))
print(torch.clamp(a, -0.5, 0.5))#clamp_0.5보다 크면 0.5로 작으면 -0.5로

#trigonometric functions and their inverses
angles=torch.tensor([0, math.pi/4, math.pi/2, 3*math.pi/4])
sines=torch.sin(angles)
inverses=torch.asin(sines)#inverse of sin
print('\nSine and arcsine: ')
print(angles)
print(sines)
print(inverses)

#bitwise operations***
print('\nBitwise XOR: ')
b=torch.tensor([1,5,11])
c=torch.tensor([2,7,10])
print(torch.bitwise_xor(b,c))

#comparisons
print('\nBroadcasted, element-wise equality comparision:')
d=torch.tensor([[1., 2.], [3., 4.]])
e=torch.ones(1,2)#satisfy condition of broadcasting!
print(torch.eq(d,e))#returns a tensor of type bool

#reductions
print('\nReduction ops:')
print(torch.max(d))
print(torch.max(d).item())#item like unboxing of JAVA
print(torch.mean(d))
print(torch.std(d))#표준편차. 분산(편차제곱 평균) 루트
print(torch.prod(d))#multiple all element
print(torch.unique(torch.tensor([1,2,1,2,1,2])))#return unique tensor of input tensor not d

#vector and linear algebra operations
v1=torch.tensor([1.,0.,0.])#x unit vector
v2=torch.tensor([0.,1.,0.])#y unit vector
m1=torch.rand(2,2)#random matrix
m2=torch.tensor([[3.,0.], [0.,3.]])#three times identity matrix

print('\nVectors & Matrices:')
print(torch.cross(v2,v1))
print(m1)
m3=torch.matmul(m1,m2)
print(m3)
print(torch.svd(m3),end='\n\n')

#altering tensor in place. like similar concept of emplace_back than push_back on c++. like shallow copy by simple concept.
a=torch.tensor([0, math.pi/4, math.pi/2, 3*math.pi/4])
print('a:')
print(a)
print(torch.sin(a))#new allocation
print(a)#not changed

b=torch.tensor([0, math.pi/4, math.pi/2, 3*math.pi/4])
print('\nb:')
print(b)
print(torch.sin_(b))#use initial memory
print(b)#changed

 #similar calculation in arthmetic operations
a=torch.ones(2,2)
b=torch.rand(2,2)

print('Before:')
print(a)
print(b)
print('\nAfter adding:')
print(a.add_(b))#altering in place
print(a)
print(b)
print('\nAfter multiplying')
print(b.mul_(b))
print(b,end='\n\n')

 #silar calculation by definition of output
a=torch.rand(2,2)
b=torch.rand(2,2)
c=torch.zeros(2,2)
old_id=id(c)

print(c)
d=torch.matmul(a,b,out=c)
print(c)

assert c is d#same with c d
assert id(c), old_id#same id; c isn't reallocated.

torch.rand(2,2,out=c)
print(c,end='\n\n')
assert id(c), old_id#same id; c isn't reallocated.

#tensor copy
 #problem
a=torch.ones(2,2)
b=a#shallow copy! not deep copy
a[0][1]=561
print(b)

 #solution by clone()
a=torch.ones(2,2)
b=a.clone()#deep copy!

assert b is not a#id is not same; difference space.
print(torch.eq(a,b))#value is same

a[0][1]=561
print(b,end='\n\n')

 #WARNNING_autograd option is keeping on deepcopy. this example is solution on modifying that option by .detach()*******
a=torch.rand(2,2, requires_grad=True)#turn on autograd_warnning point on clone()
print(a)

b=a.clone()
print(b, b.requires_grad)#tracing!

c=a.detach().clone()#work as requires_grad=True. detach seperate tensor on log of calculation
print(c, c.requires_grad)#not tracing. seperated!

print(a)

#move to GPU(CUDA_Compute Unified Device Architecture)
if torch.cuda.is_available():
    print('We habe a GPU!')
    gpu_rand=torch.rand(2,2,device='cuda')#for use CUDA, we have to assign device.
    print(gpu_rand)
else:
    print('Sorry, CPU only.')

if torch.cuda.is_available():#for compatibility
    my_device=torch.device('cuda')
else:
    my_device=torch.device('cpu')
print('Device: {}'.format(my_device))
x=torch.rand(2,2, device=my_device)
print(x)

 #move other device by .to()
y=torch.rand(2,2)
y=y.to(my_device)

 #calculation of two data that's location is different makes runtime error
try:
    x=torch.rand(2,2)
    y=torch.rand(2,2, device='gpu')
    z=x+y
except RuntimeError as e:
    print(e, end='\n\n')

#modify shape of tensor
a=torch.rand(3,226,226)
b=a.unsqueeze(0)#add dimention_CREATE. argument is location will be added additiaonal dimention
print(a.shape)
print(b.shape)

c=torch.rand(1,1,1,1,1)#just multi dimention tensor..argument is shape of tensor
print(c, end='\n\n')

a=torch.rand(1,20)
print(a.shape)
print(a)

b=a.squeeze(0)#reshaped to shape(20)!_DELETE
print(b.shape)
print(b)

c=torch.rand(2,2)
print(c.shape)

d=c.squeeze(0)#not reshaped! only we can us degree in range(1)
print(d.shape, end='\n\n')

 #we can use algo squeeze & unsqueeze on broadcasting!
a=torch.ones(4,3,2)
c=a*torch.rand(3,1)
print(c)

a=torch.ones(4,3,2)
b=torch.rand(3)#imposible for broadcasting!
c=b.unsqueeze(1)#posible thanks to unsqueeze(1)! now c.shape is (3,1) that satisfy condition of broadcasting!
print(c.shape)
print(a*c, end='\n\n')#Cool!

 #version of squeeze&unsqueeze in place
batch_me=torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)#in place version!
print(batch_me.shape,end='\n\n')

 #example of reshaping as interface(return copy)
output3d=torch.rand(6,20,20)
print(output3d.shape)

input1d=output3d.reshape(6*20*20)#way1) use member function(tensor)
print(input1d.shape)

print(torch.reshape(output3d, (6*20*20,)).shape, end='\n\n')#way2) use torch.reshape. 여기서 (6*20*20,)로 표시한 이유는 pytorch가 tensor shape지정 시 튜플을 예상하는데,
#(6*20*20)으로만 사용하면 정수로 생각할 수 있으니, 아예 1요소 튜플임을 명시하기 위해서 ()괄호로 감싸고 ,로 요소1 튜플임을 명시한 것이다.

#numpy bridge
import numpy as np

numpy_array=np.ones((2,3))
print(numpy_array)

pytorch_tensor=torch.from_numpy(numpy_array)#convert to tensor
print(pytorch_tensor)

 #convert to numpy
pytorch_rand=torch.rand(2,3)
print(pytorch_rand)

numpy_rand=pytorch_rand.numpy()
print(numpy_rand,end='\n\n')

 #*****converted numpy OR converted tensor SHARE memory*****
numpy_array[1,1]=23
print(pytorch_tensor)

pytorch_rand[1,1]=17
print(numpy_rand)
