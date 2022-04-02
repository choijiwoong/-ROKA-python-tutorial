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

class LSTM:
    def __init__(self, Wx, Wh, b):#cell_state는 내부적으로 처리되기에 신경쓸 필요 없다!
        self.params=[Wx, Wh, b]#입력값이 일단 affine처리된다. f, g, i, o에 공통적으로 적용.
        self.grads=[np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]#입력값이 공통적으로 처리되는 affine의 가중치를 저장해둘예정
        self.cache=None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b=self.params
        N, H=h_prev.shape#batch_size, shape of hidden_state
        A=np.matmul(x, Wx)+np.matmul(h_prev, Wh)+b#게이트별로 공통으로 지나가는 Affine계층을 한번에 연산

        f=A[:, :H]#계산된 affine 공통계산을 각각의 게이트에서 사용가능하도록 slicing
        g=A[:, H:2*H]
        i=A[:, 2*H:3*H]
        o=A[:, 3*H:]

        f=sigmoid(f)#forget
        g=np.tanh(g)#remember
        i=sigmoid(i)#how many remember
        o=sigmoid(o)#output

        c_next=f*c_prev+g*i#이전 cell_state를 얼마나 잊을지+현재 들어온 x값을 얼마나 기억할지로 next_cell_state계산
        h_next=o*np.tanh(c_next)#cell_state와 hidden_state가 다른 부분은 tanh에 있으며, 현재 들어온 값(output)이 다음 시점에서 얼마나 중요한지를 의미한다. 즉 지금까지 계산한 cell_state가 전체적인 반영도를 의미하며 계산된 값을 tanh에 곱하여 입력과 곱하여 반영한다.

        self.cache=(x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b=self.params
        x, h_prev, c_prev, i, f, g, o, c_next=self.cache

        tanh_c_next=np.tanh(c_next)#for calc doutput
        ds=dc_next+(dh_next*o)*(1-tanh_c_next**2)#(ds는 ct에서 왼쪽 +노드로 전파되는 미분값!)분기노드기에 둘을 더한다! 뒤에는 dh*O로 tanh아래까지 미분값을 구하고, tanh미분은 (1-y^2)이기에 위에서 계산한 실제 자신의 값을 제곱하고 1에서 뺀다.
        dc_prev=ds*f#바로 f와 곱해서 이전 timestep의 dc구해버림.

        di=ds*g#위에 dc구하며 나온 찌끄래기로 바로 di계산!
        df=ds*c_prev#술술풀린다
        do=dh_next*tanh_c_next#위에 dc부터 구한게 신의 한수였네
        dg=ds*i

        di*=i*(1-i)#sigmoid의 미분들
        df*=f*(1-f)#즉, 각 ifog에 대하여 알아서 미분들하시고
        do*=o*(1-o)
        dg*=(1-g**2)#tanh의 미분

        dA=np.hstack((df, dg, di, do))#forwarding시 slicing됐던것들이니 hstack으로 가로방향 concatenate

        dWh=np.dot(h_prev.T, dA)#일반 matmul 미분
        dWx=np.dot(x.T, dA)
        db=dA.sum(axis=0)#그냥 흘리기

        self.grads[0][...]=dWx#총 도출된 앞부분의 통일된 affine 가중치 멤버함수로 저장. 이 값이 왜 중요할지는 아직 잘 모르겠음.
        self.grads[1][...]=dWh
        self.grads[2][...]=db

        dx=np.dot(dA, Wx.T)#실제 반환해야하는 값 마저 계산
        dh_prev=np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev#LSTM입력으로 들어온 x, c, h에 대한 미분 반환

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params=[Wx, Wh, b]
        self.grads=[np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers=None
        self.h, self.c=None, None
        self.dh=None#입력받은 previous timestep의 hidden_state미분
        self.stateful=stateful#여러 timesteps의 hidden_state와 cell_state를 

    def forward(self, xs):
        Wx, Wh, b=self.params
        N, T, D=xs.shape#batch_size, 개수, dimention of input
        H=Wh.shape[0]#hidden-size

        self.layers=[]
        hs=np.empty((N,T,H), dtype='f')#각 time_step별 hs를 저장하기 위함

        if not self.stateful or self.h is None:#cell_state와 hidden_state에 대하여 초기화가 필요하면 초기화
            self.h=np.zeros((N,H), dtype='f')
        if not self.stateful or self.c is None:
            self.c=np.zeros((N,H), dtype='f')

        for t in range(T):#개수(timesteps)에 대하여
            layer=LSTM(*self.params)
            self.h, self.c=layer.forward(xs[:,t,:], self.h, self.c)#데이터와 states전달
            hs[:,t,:]=self.h#얻은 hidden_state저장

            self.layers.append(layer)
            
        return hs#최종 hidden-state반환

    def backward(self, dhs):
        Wx, Wh, b=self.params
        N, T, H=dhs.shape
        D=Wx.shape[0]

        dxs=np.empty((N,T,D), dtype='f')
        dh, dc=0,0

        grads=[0,0,0]
        for t in reversed(range(T)):
            layer=self.layers[t]
            dx, dh, dc=layer.backward(dhs[:, t, :]+dh, dc)
            dxs[:,t,:]=dx#각 시점의 dx저장
            for i, grad in enumerate(layer.grads):
                grads[i]+=grad

        for i, grad in enumerate(grads):
            self.grads[i][...]=grad#forwarding하며 각 LSTM에 저장된 dWx, dWh, db를 업데이트
        self.dh=dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c=h, c#훈련시에는 각 h와 c를 사용

    def reset_state(self):
        self.h, self.c=None, None
