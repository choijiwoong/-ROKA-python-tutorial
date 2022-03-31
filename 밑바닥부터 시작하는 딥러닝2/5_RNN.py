import numpy as np
from time_layers import *

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params=[Wx, Wh, b]
        self.grads=[np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache=None

    def forward(self, x, h_prev):
        Wx, Wh, b=self.params
        t=np.matmul(h_prev, Wh)+np.matmul(x, Wx)+b#이전 hidden_state와 곱해지는 가중치, 입력값과 곱해지는 가중치
        h_next=np.tanh(t)#를 tanh지나게 한다.

        self.cache=(x, h_prev, h_next)
        return h_next#이때 h_next를 반환하는데 이 값이 다음 timestep으로 전달되는 hidden_state이자 해당 timestep에서의 RNN셀의 출력값이다.

    def backward(self, dh_next):
        Wx, Wh, b=self.params
        x, h_prev, h_next=self.cache

        dt=dh_next*(1-h_next**2)#하이퍼 탄젠트의 미분일 뿐이다.
        db=np.sum(dt, axis=0)#b가 scalar값인데 dt값은 matrix라 그냥 sum한 거지 의미 없음. +노드라 dt값을 흘릴 뿐임.
        dWh=np.matmul(h_prev.T, dt)#여기부턴 자명한 곱 노드의 미분값
        dh_prev=np.matmul(dt, Wh.T)
        dWx=np.matmul(x.T, dt)
        dx=np.matmul(dt, Wx.T)

        self.grads[0][...]=dWx#미분값을 grads에 저장
        self.grads[1][...]=dWh
        self.grads[2][...]=db

        return dx, dh_prev#후, x에 대한 미분값과 hidden_state(전 timestep)에 대한 미분값을 반환

class TimeRNN:#RNN계층을 여러개 연결하며, 이들을 연결하기 위한 hidden_state는 인스턴스 변수로 유지한다. 이 클래스의 의미는 timestep별로 연결되는 RNN을 구현하기 위함이다.
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params=[Wx, Wh, b]
        self.grads=[np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers=None

        self.h, self.dh=None, None#
        self.stateful=stateful#씨발...빨래 안마른 냄새나서 바지인가 맡으려고 당기다가 바지 놓치면서 눈을 가격해버렸네...실명하는거아니누..라고하기엔 너무나도 잘보이고 아프기만 하다..
        #stateful은 TimeRNN계층 간 은닉상태를 인계하는 역활을 하며, 순전파를 끊지 않고 전파하며, 역전파만 끊는다.
    def set_state(self, h):
        self.h=h

    def reset_state(self):
        self.h=None

    def forward(self, xs):#xs는 sequence data
        Wx, Wh, b=self.params
        N, T, D=xs.shape#batch, 개수, dimention of input vector
        D, H=Wx.shape#입력 가중치의 shape

        self.layers=[]
        hs=np.empty((N,T,H), dtype='f')#빈 행렬(hidden-state저장목적)

        if not self.stateful or self.h is None:#h가 None이거나 stateful!=True면, h 초기화
            self.h=np.zeros((N,H), dtype='f')

        for t in range(T):#배치 크기만큼의
            layer=RNN(*self.params)#RNN레이어 형성하는데 사실상 하나의 RNN이기에 하나의 params사용.
            self.h=layer.forward(xs[:, t, :], self.h)#xs에서 해당하는 배치 데이터에 대하여 forward후 hidden_state저장
            hs[:, t, :]=self.h#hidden_state저장
            self.layers.append(layer)

        return hs#출력값(hidden_state)반환

    def backward(self, dhs):
        Wx, Wh, b=self.params
        N, T, H=dhs.shape
        D, H=Wx.shape

        dxs=np.empty((N,T,D), dtype='f')#크게 입력 데이터에 대한 미분과
        dh=0#hidden_state에 대한 미분을 계산하면 됨.
        grads=[0,0,0]#dhs(출력값에 대한 들어오는 미분), dh(prev_hidden_state), dxs(입력에 대한 미분)
        for t in reversed(range(T)):
            layer=self.layers[t]
            dx, dh=layer.backward(dhs[:, t, :]+dh)#(합산된 기울기)각 레이어에 backward를 수행한 결과를(참고로 마지막 for의 dh가 최종 dh..가장 최초 hs니)
            dxs[:, t, :]=dx#해당 배치에 해당하는 dxs에 저장.
            for i, grad in enumerate(layer.grads):#각 timestep의 rnn grad를 저장
                grads[i]+=grad

        for i, grad in enumerate(grads):
            self.grads[i][...]=grad#각 RNN기울기를 self.grads에 저장
        self.dh=dh

        return dxs#dxs리턴. 각 기울기들은 맴버 변수에 저장됨.

#1. RNNLM의 구현
class SimpleRnnlm:
    def __init__(self, vocab_size ,wordvec_size, hidden_size):#for embedding & affine
        V, D, H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn
        
        #가중치 초기화(Xaiver)
        embed_W=(rn(V,D)/100).astype('f')#임베딩 가중
        rnn_Wx=(rn(D,H)/np.sqrt(D)).astype('f')#RNN 입력 가중
        rnn_Wh=(rn(H,H)/np.sqrt(H)).astype('f')#RNN 은닉 가중
        rnn_b=np.zeros(H).astype('f')#RNN 편향
        affine_W=(rn(H,V)/np.sqrt(H)).astype('f')#아핀 가중
        affine_b=np.zeros(V).astype('f')#아핀 편향

        #계층 생성
        self.layers=[#emb->RNN->Affine
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer=TimeSoftmaxWithLoss()#loss
        self.rnn_layer=self.layers[1]#RNN만 따로 저장 for using set_statefule & reset_stateful function

        self.params, self.grads=[], []
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs=layer.forward(xs)
        loss=self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout=self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()#별도로 저장한 rnn_layer의 reset_state호출

#2. RNNLM의 학습
import matplotlib.pyplot as plt
from dataset import ptb
from optimizers import SGD

#hyper parameter
batch_size=10
wordvec_size=100
hidden_size=100
time_size=5
lr=0.1
max_epoch=100

#read data
corpus, word_to_id, id_to_word=ptb.load_data('train')
corpus_size=1000
corpus=corpus[:corpus_size]
vocab_size=int(max(corpus)+1)

xs=corpus[:-1]#(입력)맨 마지막 단어를 제외한 corpus Hmm...아 하 ! abcde 가 총데이터면 입력이 abcd이고 레이블이 bcde구나 시계열이니!
ts=corpus[1:]#(출력)맨 첫단어를 제외한 corpus
data_size=len(xs)
print('corpus크기: ', corpus_size, ', 어휘 수: ', vocab_size)

max_iters=data_size//(batch_size*time_size)#for train
time_idx=0
total_loss=0
loss_count=0
ppl_list=[]#perplexity

model=SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer=SGD(lr)
"""
jump=(corpus_size-1)//batch_size#총 몇번의 batch calculation
offsets=[i*jump for i in range(batch_size)]#jump를 고려한 배치당 index(offset)저장

for epoch in range(max_epoch):#epoch반복
    for iter in range(max_iters):#최대 loop횟수에 대하여
        batch_x=np.empty((batch_size, time_size), dtype='i')#batch를 위한 공간 할당
        batch_t=np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):#시계열
            for i, offset in enumerate(offsets):#오프셋
                batch_x[i,t]=xs[(offset+time_idx)%data_size]#시계열 고려, 오프셋활용하여 해당 배치 데이터들 저장.
                batch_t[i,t]=ts[(offset+time_idx)%data_size]
            time_idx+=1

        loss=model.forward(batch_x, batch_t)#배치들을 모델에 전달
        model.backward()#역전파
        optimizer.update(model.params, model.grads)#매개변수 업데이트
        total_loss+=loss
        loss_count+=1

    ppl=np.exp(total_loss/loss_count)
    print('| epoch %d | perplexity %.2f'%(epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count=0, 0
"""
from trainer import RnnlmTrainer

trainer=RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
