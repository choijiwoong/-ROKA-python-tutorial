import numpy as np
from layers import *
from functions import *

class TimeEmbedding:
    def __init__(self, W):
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.layers=None
        self.W=W

    def forward(self, xs):
        N, T=xs.shape#batch, timesteps
        V, D=self.W.shape#vocab_size, embedding_size

        out=np.empty((N,T,D), dtype='f')
        self.layers=[]

        for t in range(T):#각 timestep별
            layer=Embedding(self.W)
            out[:,t,:]=layer.forward(xs[:,t])#emb결과를 out에 저장
            self.layers.append(layer)#for backpropagation

        return out

    def backward(self, dout):
        N, T, D=dout.shape#배치, 시간, 입력차원크기 들어옴 (왜냐면 RNN에 사용해야하니..)

        grad=0
        for t in range(T):
            layer=self.layers[t]
            layer.backward(dout[:,t,:])
            grad+=layer.grads[0]#어차피 더해서 순서는 상관x

        self.grads[0][...]=grad
        return None

class TimeAffine:
    def __init__(self, W, b):
        self.params=[W,b]
        self.grads=[np.zeros_like(W), np.zeros_like(b)]
        self.x=None

    def forward(self, x):
        N, T, D=x.shape
        W, b=self.params

        rx=x.reshape(N*T, -1)#입력값을 flatten(배치, timestep만. without dimention of input)
        out=np.dot(rx, W)+b#dot
        self.x=x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x=self.x
        N, T, D=x.shape
        W, b=self.params

        dout=dout.reshape(N*T, -1)
        rx=x.reshape(N*T, -1)

        db=np.sum(dout, axis=0)#흘려보내는거라 그냥 dout을 받는데 scalar다보니 sum해주는거
        dW=np.dot(rx.T, dout)#Wx니 x.T
        dx=np.dot(dout, W.T)#W.T
        
        dx=dx.reshape(*x.shape)#just reshaping

        self.grads[0][...]=dW#grads에 저장 후
        self.grads[1][...]=db

        return dx#dx리턴

class TimeSoftmaxWithLoss:#보강필요
    def __init__(self):
        self.params, self.grads=[], []
        self.cache=None
        self.ignore_label=-1

    def forward(self, xs, ts):
        N, T, V=xs.shape

        if ts.ndim==3:#label이 one-hot인 경우 처리
            ts=ts.argmax(axis=2)

        mask=(ts!=self.ignore_label)#음 어떤 이유던 간에 마스킹하고 싶은 값을 -1로 하여 넣으면 마스크해주는 부가적인 기능인듯

        xs=xs.reshape(N*T, V)
        ts=ts.reshape(N*T)
        mask=mask.reshape(N*T)

        ys=softmax(xs)#softmax with
        ls=np.log(ys[np.arange(N*T), ts])#crossentropy
        ls*=mask
        loss=-np.sum(ls)#error
        loss/=mask.sum()#나머진 마스킹 처리

        self.cache=(ts, ys, mask, (N,T,V))#label, predict, mask, (batch, Timesteps, Vector_dimention)
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N,T,V)=self.cache
        dx=ys#pred
        dx[np.arange(N*T), ts]-=1#모든 dx에 대하여 -1(softmaxwithloss미분은 y-t로 y에서 t에 해당하는 부분을 -1)
        dx*=dout#사실 이것도 잘 모르게썽
        dx/=mask.sum()#마스크처리 사실 잘 모르게썽
        dx*=mask[:, np.newaxis]#ignore_label에 해당하는 데이터의 기울기(dx)를 0으로 만드는거라는데..

        dx=dx.reshape((N,T,V))

        return dx

class TimeDropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads=[], []
        self.dropout_ratio=dropout_ratio
        self.mask=None
        self.train_flg=True

    def forward(self, xs):
        if self.train_flg:
            flg=np.random.rand(*xs.shape)>self.dropout_ratio#ratio이상만
            scale=1/(1.0-self.dropout_ratio)#원상복구를 위한 scale
            self.mask=flg.astype(np.float32)*scale#본래의 크기

            return xs*self.mask
        else:
            return xs

    def backward(self, dout):
        return dout*self.mask
