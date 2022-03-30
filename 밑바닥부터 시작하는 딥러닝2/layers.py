from functions import softmax, cross_entropy_error
import numpy as np

class MatMul:
    def __init__(self, W):#입력받은 가중치를
        self.params=[W]#리스트로 저장하며
        self.grads=[np.zeros_like(W)]
        self.x=None#연산 대상

    def forward(self, x):
        W, =self.params
        out=np.matmul(x, W)#forward시 저장되어있는 params에서 가중치를 가져와서 곱하고 
        self.x=x#backward를 위해 인자를 저장한다.
        return out

    def backward(self, dout):
        W, =self.params
        dx=np.matmul(dout, W.T)#미분값에 따라 교차 및 Transpose
        dW=np.matmul(self.x.T, dout)
        self.grads[0][...]=dW#dw값의 경우 grads에 저장해두며
        return dx#dx즉, forward 피연산자에 대한 미분값은 리턴한다.

class Affine:
    def __init__(self, W, b):
        self.params=[W, b]#계산에 사용할 값들 저장
        self.grads=[np.zeros_like(W), np.zeros_like(b)]
        self.x=None

    def forward(self, x):
        W, b=self.params
        out=np.matmul(x,W)+b
        self.x=x
        return out

    def backward(self, dout):
        W, b=self.params
        dx=np.matmul(dout, W.T)
        dW=np.matmul(self.x.T, dout)
        db=np.sum(dout, axis=0)

        self.grads[0][...]=dW
        self.grads[1][...]=db
        return dx#입력에 대한 미분값은 리턴하고 나머지는 grad에 저장.

class Softmax:
    def __init__(self):
        self.params, self.grads=[], []
        self.out=None

    def forward(self, x):
        self.out=softmax(x)#e^xi/sum(e^xn)
        return self.out

    def backward(self, dout):#이거 미분부를 잘 모르겠네..
        dx=self.out*dout
        sumdx=np.sum(dx, axis=1, keepdims=True)
        dx-=self.out*sumdx
        return dx

class Sigmoid:
    def __init__(self):
        self.params, self.grads=[], []
        self.out=None

    def forward(self, x):
        out=1/(1+np.exp(-x))#1/1+e^-x미분시 
        self.out=out#backward시 사용을 위한 저장.
        return out

    def backward(self, dout):
        dx=dout(1.0-self.out)*self.out#(1-y)y
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads=[], []
        self.loss=None
        self.y=None
        self.t=None

    def forward(self, x, t):
        self.t=t
        self.y=1/(1+np.exp(-x))#sigmoid
        self.loss=cross_entropy_error(np.c_[1-self.y, self.y], self.t)#with loss. softmax의 미분이 (1-out)out이기에 바로 손실함수.
        return self.loss

    def backward(self, dout=1):
        batch_size=self.t.shape[0]

        dx=(self.y-self.t)*dout/batch_size#미분값은 말 그대로 오차를 의미하는 y-t
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads=[], []
        self.y=None
        self.t=None

    def forward(self, x, t):
        self.t=t
        self.y=softmax(x)#softmax. 어차피 with loss라 미분은 -로 나옴

        if self.t.size==self.y.size:
            self.t=self.t.argmax(axis=1)#one-hot고려

        loss=cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size=self.t.shape[0]

        dx=self.y.copy()#predict
        dx[np.arange(batch_size), self.t]-=1#오차계산
        dx*=dout
        dx/=batch_size

        return dx

class Embedding:
    def __init__(self, W):#벡터들 세팅. lookup기능이니.
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.idx=None

    def forward(self, idx):
        W, =self.params
        self.idx=idx
        out=W[idx]#해당 vector반환
        return out

    def backward(self, dout):
        dW,=self.grads
        dW[...]=0
        np.add.at(dW, self.idx, dout)#dW의 idx위치에 dout을 더한다.???이 왜 미
        return None
