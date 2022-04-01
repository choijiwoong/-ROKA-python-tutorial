"""헷갈리는 거에 대해 정리를 해보자. stateful의 역활.
stateful=True는 상태가 있다. 은닉 상태를 유지한다는 뜻으로 시계열 데이터가 아무리 길어도 순전파를 끊지 않고 유지한다는 의미이다
False는 은닉 상태를 0행렬로 초기화하여 무상태라고 한다. 이전 시각의 은닉 상태를 유지할지를 지정하는 변수로
역전파시에는 왜 영행렬로 초기화할까. 단지 Truncated BPTT를 이용해서 인데, 적당한 사이즈로 잘라 그 단위로 오차역전파를 수행하여 컴퓨팅 자원을 절약한다.
gradient vanishing의 방지이다.
"""
import numpy as np
from time_layers import *
import pickle

class Rnnlm:
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V,D,H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn

        #Xavier Initialization on weights
        embed_W=(rn(V,D)/100).astype('f')
        lstm_Wx=(rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh=(rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b=np.zeros(4*H).astype('f')#Wc는 내부적으로 처리되기에 상관X
        affine_W=(rn(H,V)/np.sqrt(H)).astype('f')
        affine_b=np.zeros(V).astype('f')#공통적으로 사용된 아핀의 가중치

        self.layers=[
            TimeEmbedding(embed_W),#단어 임베딩
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)#분류를 위함
        ]
        self.loss_layer=TimeSoftmaxWithLoss()#손실함수
        self.lstm_layer=self.layers[1]#for set_state등

        self.params, self.grads=[], []
        for layer in self.layers:#모든 레이어의 매개변수와 기울기 모음. save와 load를 위함
            self.params+=layer.params
            self.grads+=layer.grads

    def predict(self, xs):#예측을 반환
        for layer in self.layers:
            xs=layer.forward(xs)
        return xs

    def forward(self, xs, ts):#손실을 반환
        score=self.predict(xs)
        loss=self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout=self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()#연결을 끊는다

    def save_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name='Rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params=pickle.load(f)


from optimizers import SGD
from trainer import RnnlmTrainer
from utils import eval_perplexity
from dataset import ptb
"""(test)
batch_size=20
wordvec_size=100
hidden_size=100#RNN 은닉 상태의 벡터 원소 수
time_size=35#RNN펼치는 크기
lr=20.0
max_epoch=4
max_grad=0.25#gradient cliping

corpus, word_to_id, id_to_word=ptb.load_data('train')
corpus_text, _, _=ptb.load_data('test')
vocab_size=len(word_to_id)
xs=corpus[:-1]
ts=corpus[1:]

model=Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer=SGD(lr)
trainer=RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0,500))

model.reset_state()#테스트모드
ppl_test=eval_perplexity(model, corpus_test)
print('테스트 perplexity: ', ppl_test)

model.save_params()
"""
from base_model import BaseModel

class BetterRnnlm(BaseModel):#그냥 params grads할당하고 load, save함수 끝.
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V, D, H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn

        embed_W=(rn(V,D)/100).astype('f')#가중치 affine과 공유
        lstm_Wx1=(rn(D, 4*H)/np.sqrt(D)).astype('f')
        lstm_Wh1=(rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b1=np.zeros(4*H).astype('f')
        lstm_Wx2=(rn(H, 4*H)/np.sqrt(H)).astype('f')#multi layer lstm
        lstm_Wh2=(rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b2=np.zeros(4*H).astype('f')
        affine_b=np.zeros(V).astype('f')

        self.layers=[
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),#dropout
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)
        ]
        self.loss_layer=TimeSoftmaxWithLoss()
        self.lstm_layers=[self.layers[2], self.layers[4]]
        self.drop_layers=[self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads=[], []
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg=train_flg#예측시에는 dropout계층에 train_flg를 False로 하여 모든 노드 다 사용하게끔
        for layer in self.layers:
            xs=layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score=self.predict(xs, train_flg)
        loss=self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout=self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout=layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
#(test)
batch_size=20
wordvec_size=650
hidden_size=650
time_size=35
lr=20.0
max_epoch=40
max_grad=0.25
dropout=0.5

corpus, word_to_id, id_to_word=ptb.load_data('train')
corpus_val, _, _=ptb.load_data('val')
corpus_test, _, _=ptb.load_data('test')

vocab_size=len(word_to_id)
xs=corpus[:-1]
ts=corpus[1:]

model=BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer=SGD(lr)
trainer=RnnlmTrainer(model, optimizer)

best_ppl=float('inf')
for epoch in range(max_epoch):
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad)#max_epoch를 1로 설정하며 직접 max_ppl과 비교하며 fitting
    model.reset_state()
    ppl=eval_perplexity(model, corpus_val)
    print('evaluated perplexity: ', ppl)

    if best_ppl>ppl:
        best_ppl=ppl
        model.save_params()
    else:
        lr/=4.0
        optimizer.lr=lr

    model.reset_state()
    print('-'*50)
