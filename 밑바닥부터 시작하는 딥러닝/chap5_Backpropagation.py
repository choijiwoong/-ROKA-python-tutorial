#   [앞으로도 계속 임포트하여 사용할 것이기에 한번에 정리하고자 이전 챕터의 function들만 가져옴]
def stop_function(x): return np.array(x>0, dtype=np.int)

def sigmoid(x): return 1/(1+np.exp(-x))

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a=np.sum(exp_a)

    return y

def identity_function(x): return x

def relu(X): return np.maximum(0, x)

def cross_entropy_error(y, t):
    if y.ndim==1:
        t=t.reshape(1, t.size)
        y=y.reshape(1, y.size)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size

def numerical_gradient(f, x):#수치 미분
    h=1e-4
    grad=np.zeros_like(x)
    for idx in range(x.size):
        tmp_val=x[idx]

        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val#복원
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):#경사하강
    x=Init_x

    for i in range(step_num):
        grad=numerical_gradient(f, x)
        x-=lr*grad#lr*gradient만큼 --
    return x

#   [단순한 계층 구현하기]
class MulLayer:#Mul이 다수의 의미가 아니라 곱하는(Multiple) layer라는 뜻으로, 계산그래프로 보아 곱은 dout*입력의순서뒤바꾼것 이기에 그걸 아이디어로 forward와 backward를 구현한 것.
    def __init__(self):
        self.x=None
        self.y=None

    def forward(self, x, y):
        self.x=x
        self.y=y
        out=x*y#Multiple Layer이기에 둘을 곱하여 반환. 이때 x와 y를 저장한다는 것이 중요한데, 후의 backward연산에 필요하기 때문이다.

        return out

    def backward(self, dout):#이 dout은 역전파하며 입력된 미분정보로
        dx=dout*self.y#곱셈의 경우 입력의 순서의 반대를 곱하면 미분값과 같기에 이를 dx, dy로 담은다음 리턴시킨다.
        dy=dout*self.x

        return dx, dy

class AddLayer:#두 입력값을 더하는 Layer이다. 덧셈의 경우 backpropagation시 dout값을 그대로 흘려보낸다.
    def __init__(self):
        pass

    def forward(self, x, y):
        out=x+y#사용하진 않지만 가독성을 위해
        return out

    def backward(self, dout):
        dx=dout*1#MultiLayer와 달리 입력값에 의존하지 않고 dout을 그냥 흘려보낸다.
        dy=dout*1
        
        return dx, dy
"""MultiLayer, AddLayer의 사용예시.
apple=100#사과가격
apple_num=2#사과개수
orange=150#오렌지가격
orange_num=3#오렌지개수
tax=1.1#세금

#사용할 계층들의 인스턴스화(계산그래프)_나중에 backpropagation시 필요한 정보를 담고있기에 하나의 레이어를 다른 연산에 사용하면 안된다. 고로 인스턴스화하여 사용한다.
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()

#Forwarding
apple_price=mul_apple_layer.forward(apple, apple_num)
orange_price=mul_orange_layer.forward(orange, orange_num)
all_price=add_apple_orange_layer.forward(apple_price, orange_price)
price=mul_tax_layer.forward(all_price, tax)

#Backwarding
dprice=1
dall_price, dtax=mul_tax_layer.backward(dprice)#연산과정을 reverse하며 각각의 backward연산을 호출한다.
dapple_price, dorange_price=add_apple_orange_layer.backward(dall_price)#개인적으로 갑자기 드는 생각인데, 뭔가 한번 backward하면 다시 못사용하게 destroy해야할거같은 느낌..순서대로 진행하니 실수방지를 위해서랄까..
dorange, dorange_num=mul_orange_layer.backward(dorange_price)
dapple, dapple_num=mul_apple_layer.backward(dapple_price)

print('price: ', price)
print('d값들: ', dapple_num, dapple, dorange, dorange_num, dtax)
"""

#   [활성화함수계층 구현하기]
class Relu:
    def __init__(self):
        self.mask=None

    def forward(self, x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0#x가 0이하면 0으로 만들고 아니면 냅둬서 반환.

        return out

    def backward(self, dout):#Relu는 음수일때 0, 양수일때 x이기에 미분값도 0과 1이다.
        dout[self.mask]=0#음수인 부분은 0으로 만들고 나머지는 dout그대로를 유지하게 한 뒤 반환.
        dx=dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out=None

    def forward(self, x):
        out=1/(1+np.exp(-x))#sigmoid연산
        self.out=out#Backward시 사용하기에 저장.

        return out

    def backward(self, dout):#이 짜릿한 유도는 직접 사진찍어 올리고 싶지만 국방모바일보안. 아쉬운대로 참고사진.
        dx=dout*(1.0-self.out)*self.out#Sigmoid함수 각 과정을 계산그래프로 만들어서 미분 계속 하면 y(1-y)나옴(y는 sigmoid출력값)


#   [Affine/Softmax계층 구현하기]
class Affine:
    def __init__(self, W, b):
        self.W=W
        self.b=b
        self.x=None#for backward
        self.dW=None#too
        self.db=None#too

    def forward(self, x):
        self.x=x#for backward
        out=np.dot(x, self.W)+self.b#x와 W의 matmul

        return out

    def backward(self, dout):#Affine의 backward는 matmul(dot)의 미분이 필요한데, 곱하는 것과 마찬가지로 뒤바뀌나 각 matrix의 shape도 뒤바뀌어 Transpose해준다.
        dx=np.dot(dout, self.W.T)#dx에 dout matmul W인데 W마저도 뒤바뀌어 Transpose버전(뭐 np.array일테니... Transpose..)
        self.dw=np.dot(self.x.T, dout)
        self.db=np.sum(dout, axis=0)#그냥 더하기였기에 dout그대로 내보내는데, 배치를 고려 Matrix연산이다보니 shape만 바꾼다. ex) dout이 (N,3)이면 db는 (3)으로

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None

    def forward(self, x, t):
        self.t=t
        self.y=softmax(y)#softmax와
        self.loss=cross_entropy_error(self.y, self.t)#cross entropy 손실함수 계산을 한번에 하는 이유는
        return self.loss

    def backward(self, dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size#복잡한 softmax에 손실함수를 곱하면 미분이 존나게 간단하게 나오게끔 softmax용 cross_entropy_loss function을 설계했기 때문이다.
        #단순히 빼면된다.
        return dx
    
#Backpropagation을 적용한 신경망의 예시
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.rand(hidden_size, output_size)
        self.params['b2']=np.zeros(output_size)

        #계층 준비
        self.layers=OrderedDict()
        self.layers['Affine1']=Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'], self.params['b2'])
        self.lastLayer=SoftmaxWithLoss()#non-ReLU!

    def predict(self, x):
        for layer in self.layers.values():
            x=layer.forward(x)

        return x

    def loss(self, x, t):
        y=self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)#예측값 index 게또
        if t.ndim!=1:#필요하다면 label 데이터에 대해서도 꼴 맞춰주기
            t=np.argmax(t, axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])#개수(평균을 위함)
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W=lambda W: self.loss(x, t)#손실함수를 람다로 생성 (수치미분 준비)

        grads={}
        grads['W1']=numerical_gradient(loss_W, self.params['W1'])#손실항수를 W1기준 편미분
        grads['b1']=numerical_gradient(loss_W, self.params['b1'])
        grads['W2']=numerical_gradient(loss_W, self.params['W2'])
        grads['b2']=numerical_gradient(loss_W, self.params['b2'])

    def gradient(self, x, t):
        #순전파
        self.loss(x, t)

        #역전파
        dout=1
        dout=self.lastLayer.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)#backward를 단순히 역순으로 호출하고 각각의 미분값들을 grads에 저장하여 리턴.

        #결과 저장
        grads={}
        grads['W1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['W2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db

        return grads

"""오차역전파법을 사용한 학습 구현하기
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test)=load_mnist(normalize=True, one_hot_label=True)#Data get
network=TwoLayerNet(input_size=784, hidden_size=50, output_size=10)#Make Two Layer Net that can backpropagate

iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1

train_loss_list=[]
train_acc_list=[]
test_acc_list=[]

iter_per_epoch=max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask=np.random.choice(train_size, batch_size)#batch_size만큼 랜덤으로 인덱스 추출.
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    grad=network.gradient(x_batch, t_batch)#미분값 저장

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key]-=learning_rate*grad[key]#경사하강(가중치 조율)

    loss=network.loss(x_batch, t_batch)    train_loss_list.append(loss)

    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train, t_train)
        test_acc=network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
"""
