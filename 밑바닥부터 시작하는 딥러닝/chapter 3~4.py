import numpy as np
import sys, os
sys.path.append(os.pardir)#부모 디렉토리 파일을 가져올 수 있게 설정
from dataset.mnist import load_mnist
    
    #[기본적인 도구의 구현]
def step_function(x):#0과 1
    #y=x>0
    #return y.astype(np.int)#numpy형식으로 반환
    return np.array(x>0, dtype=np.int)
def sigmoid(x):#연속적인 값
    return 1/(1+np.exp(-x))
def softmax(a):#전체 exponential중 특정 exeponential
    c=np.max(a)#overflow를 막기 위함(큰값끼리의 나눗셈은 불안정하다)
    exp_a=np.exp(a-c)#빼줌.
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y
def identity_function(x):
    return x
def relu(X):
    return np.maximum(0,x)
def cross_entropy_error(y, t):
    if y.ndim==1:
        t=t.reshape(1, t.size)
        y=y.reshape(1, y.size)
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size
def numerical_gradient(f, x):#수치미분
    h=1e-4
    grad=np.zeros_like(x)
    for idx in range(x.size):
        tmp_val=x[idx]

        x[idx]=tmp_val+h
        fxh1=f(x)

        x[idx]=tmp_val-h
        fxh2=f(x)

        grad[idx]=(fxh1-fxh2)/(2*h)#grad에 미분값 저장
        x[idx]=tmp_val#복원.
    return grad#편미분 행렬 반환.
def gradient_descent(f, init_x, lr=0.01, step_num=100):#경사하강
    x=init_x

    for i in range(step_num):
        grad=numerical_gradient(f, x)
        x-=lr*grad
    return x


    """[신경망의 추론처리]
def get_data():
    (x_train, t_train), (x_test, t_test)=load_mnist(normalize=True. flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    #with open('sample_weight.pkl", 'rb') as f:
    #   network=pickle.load(f)
    #return network
    network={}
    network['W1']=np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])#2,3
    network['b1']=np.array([0.1, 0.2, 0.3])
    network['W2']=np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])#3,2
    network['b2']=np.array([0.1, 0.2])
    network['W3']=np.array([[0.1, 0.3], [0.2, 0.4]])#2,2
    network['b3']=np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3=network['W1'], network['W2'], network['W3']
    b1, b2, b3=network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1)+b1#1,2 * 2,3=1,3
    z1=sigmoid(a1)
    a2=np.dot(z1, W2)+b2#1,3 * 3,2=1,2
    z2=sigmoid(a2)
    a3=np.dot(z2, W3)+b3#1,2 * 2,2=1,2
    y=identity_function(a3)

    return y

x, t= get_data()
network=init_network()

batch_size=100#배치for 효율적 처리, 버스부하감소, 순수계산비율 증가
accuracy_cnt=0

for i in range(0, len(x), batch_size):#step
    x_batch=x[i:i+batch_size]
    y_batch=predict(network, x_batch)
    p=np.argmax(y_batch, axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])
print("Accuracy: "+str(float(accuracy_cnt)/len(x)))"""


    #[Simple Net]
class simpleNet:
    def __init__(self):
        self.W=np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y, t)
        return loss
net=simpleNet()
print('SimpleNet의 가중치 매개변수: \n', net.W)

x=np.array([0.6, 0.9])
p=net.predict(x)
print(x,'에 대한 simpleNet의 예측 값: ', p)
print('예측값의 최대값 인덱스: ', np.argmax(p))

t=np.array([0,0,1])#Label
print('예측값의 오차값: ', net.loss(x,t))

def f(W):#dummy function
    return net.loss(x, t)
#dW=numerical_gradient(f, net.W)#dummpy function과 net의 weight를 수치미분함수에 넣는다.
#print('\nnumerical gradient를 이용한 가중치 미분값: \n', dW)#오류뜨뮤ㅠ


    #[2층 신경망 클래스 구현하기 with min-batch]
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params={}
        self.params['W1']=weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2']=np.zeros(output_size)

    def predict(self, x):
        W1, W2=self.params['W1'], self.params['W2']
        b1, b2=self.params['b1'], self.params['b2']

        a1=np.dot(x, W1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,W2)+b2
        y=softmax(a2)

        return y

    def loss(self, x, t):
        y=self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        t=np.argmax(t, axis=1)

        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W=lambda W: self.loss(x,t)#dummy

        grads={}
        grads['W1']=numerical_gradient(loss_W, self.params['W1'])
        grads['b1']=numerical_gradient(loss_W, self.params['b1'])
        grads['W2']=numerical_gradient(loss_W, self.params['W2'])
        grads['b2']=numerical_gradient(loss_W, self.params['b2'])

        return grads
net=TwoLayerNet(inputs_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)

 #미니배치 학습 구현 + test set구현
(x_train, t_train), (x_test, t_test)=load_mnist(normalize=True, one_hot_label=True)

iters_num=10000
train_size=x_train.shape[0]
batch_size=100
learning_rate=0.1

train_loss_list=[]#loss값 저장용
train_acc_list=[]
test_acc_list=[]

iter_per_epoch=max(train_size/batch_size, 1)#1epoch당 반복수

network=TwoLayerNet(input_size=28*28, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask=np.random.choice(train_size, batch_size)#미니배치 획득
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]

    grad=network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key]-=learning_rate*grad[key]

    loss=network.liss(x_batch, t_batch)
    train_loss_list.append(loss)

    #1epoch당 정확도 계산
    if i%iter_per_epoch==0:
        train_acc=network.accuracy(x_train, t_train)
        test_acc=network.accuracy(x_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc : ', train_acc, ', ', test_acc)
