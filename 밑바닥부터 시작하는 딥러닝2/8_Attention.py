import numpy as np
from layers import Softmax
"""
seq2seq를 개선하는 reverse, peeky를 넘어서서 encoder의 모든 hidden_state를 반영하게끔
그들의 출력에 가중합하여 모든 hs를 반영하는 context 벡터를 만들기 위함이며 Mini-Batch를 고려하여 N을 추가한다
 첫번째로, context벡터를 만들기 위한 과정은 아래와 같다."""
N, T, H=10, 5, 4#batch_size, timesteps, hidden_size
hs=np.random.randn(N, T, H)#배치별 단어각각의 은닉 상태값들
a=np.random.randn(N, T)#각각의 은닉상태에 곱할 가중치값
ar=a.reshape(N, T, 1).repeat(H, axis=2)#각 은닉state에 곱하기 위해 repeat 즉, 브로드캐스팅(명시적)

t=hs*ar#인코더의 은닉상태들과 브로드캐스팅된 가중치 곱셈
print(t.shape)

c=np.sum(t, axis=1)#합
print(c.shape,'\n')

class WeightSum:#위의 과정에서 가중합을 간편화 하기 위한 함수.
    def __init__(self):
        self.params, self.grads=[], []
        self.cache=None

    def forward(self, hs, a):
        N, T, H=hs.shape

        ar=a.reshape(N, T, 1).repeat(H, axis=2)
        t=hs*ar
        c=np.sum(t, axis=1)

        self.cache=(hs, ar)
        return c

    def backward(self, dc):
        hs, ar=self.cache
        N,T,H=hs.shape

        dt=dc.reshape(N,1,H).repeat(T, axis=1)
        dar=dt*hs
        dhs=dt*ar
        da=np.sum(dar, axis=2)

        return dhs, da

"""인코더의 hs에 곱할 a 즉 가중치를 구하기 위해서는 우선 decoder의 input에 따른 hs를 참고한다"""
N, T, H=10,5,4
hs=np.random.randn(N,T,H)#encoder's hs
h=np.random.randn(N,H)#디코더의 입력을 통해 나온 은닉 상태값
hr=h.reshape(N,1,H)#브로드캐스팅 for 유사도계산

t=hs*hr#인코더의 hidden_state들과 디코더의 hidden_state의 유사도 계산(dot-product)
print(t.shape)#10 5 4

s=np.sum(t, axis=2)#인코더의 hidden_state별 유사도 점수
print(s.shape)#10 5

softmax=Softmax()
a=softmax.forward(s)#확률분포화
print(a.shape)

class AttentionWeight:#인코더의 hidden_state들과 decoder의 output hidden_state유사도를 계산할 dot-product attention weight함수
    def __init__(self):
        self.params, self.grads=[], []
        self.softmax=Softmax()#확률화 용
        self.cache=None

    def forward(self, hs, h):#인코더의 은닉상태들과 디코더의 출력 은닉상태입력
        N, T, H=hs.shape

        hr=h.reshape(N, 1, H).repeat(T, axis=1)#h의 명시적 브로드 캐스팅
        t=hs*hr#dot-product
        s=np.sum(t, axis=2)#scoring
        a=self.softmax.forward(s)#to distribution

        self.cache=(hs, hr)#for backwarding
        return a

    def backward(self, da):
        hs, hr=self.cache
        N,T,H=hs.shape

        ds=self.softmax.backward(da)
        dt=ds.reshape(N, T, 1).repeat(H, axis=2)#합의 미분은 리핏
        dhs=dt*hr
        dhr=dt*hs
        dh=np.sum(dhr, axis=1)

        return dhs, dh#입력으로 사용된 hs(인코더)와 h(디코더)에 대한 미분값 반환
"""
정리하면, 모든 시점의 인코더 은닉상태를 참고하기 위해 모든 시점의 은닉상태값들에 특정 가중치를 가중합하여 context vector를 만드는데,
이때의 가중치는 인코더의 모든 시점 은닉상태와 디코더의 출력은닉상태값의 유사도(dot-product)를 곱하여 계산한 score기반 distribution값이다
이들을 각각 WeightSum과 AttentionWeight로 정리하였는데 이 둘을 엮어 Attention클래스를 정의한다"""
class Attention:
    def __init__(self):
        self.params, self.grads=[], []
        self.attention_weight_layer=AttentionWeight()
        self.weight_sum_layer=WeightSum()
        self.attention_weight=None

    def forward(self, hs, h):#인코더의 은닉상태들과 디코더의 출력은닉상태를 받아
        a=self.attention_weight_layer.forward(hs, h)#attention_weight를 계산한뒤
        out=self.weight_sum_layer.forward(hs, a)#인코더의 은닉상태들에 가중합하여 context벡터를 얻는다.
        self.attention_weight=a
        return out

    def backward(self, dout):#forward시 입력값인 hs와 h의 미분을 반환하면되며 각각의 backward를 차례로 수행하면된다.
        dhs0, da=self.weight_sum_layer.backward(dout)#인코더의 hs, weight에 대한 미분
        dhs1, dh=self.attention_weight_layer.backward(da)#인코더의 hs, 디코더h에 대한 미분
        dhs=dhs0+dhs1#hs2개가 사용되었기에 sum(분기의 미분은 합)
        return dhs, dh
"""이 어텐션 계층을 기존의 Rnnlm의 LSTM과 Affine사이에 끼워넣으면 된다. 시계열 어텐션의 구현은 아래와 같다"""
class TimeAttention:
    def __init__(self):
        self.params, self.grads=[], []
        self.layers=None#시계열 처리를 위한 어텐션 배열 
        self.attention_weights=None#가중치는 인코더 hs과 디코더 timestep별 h를 사용하기에 시계열마다 변화한다.

    def forward(self, hs_enc, hs_dec):
        N, T, H=hs_dec.shape#디코더
        out=np.empty_like(hs_dec)#디코더의 timestep별 모든 attention결과를 저장예정(h?)
        self.layers=[]
        self.attention_weights=[]

        for t in range(T):
            layer=Attention()
            out[:,t,:]=layer.forward(hs_enc, hs_dec[:,t,:])#timestep별 디코더 입력으로 attention
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)#decoder timestep별 weight저장
        return out

    def backward(self, dout):
        N,T,H=dout.shape
        dhs_enc=0#forward입력에 대한 미분계산예정
        dhs_dec=np.empty_like(dout)

        for t in range(T):#timestep별로
            layer=self.layers[t]
            dhs, dh=layer.backward(dout[:, t, :])#backwarding후
            dhs_enc+=dhs#dhs_enc에 가산(하나의 인코더 hs들이 여러 시계열decoder에 repeat되었으니 sum)
            dhs_dec[:,t,:]=dh#dec에 저장

        return dhs_enc, dhs_dec
"""시계열 어텐션으로 seq2seq를 구현하기 이전에, 정리차. attention은 기존의 seq2seq모델 encoder-decoder로 구성된 시계열 처리 모델에서
인코더의 hs가 아닌 최종 h를 context vector로 압축의 문제가 있었는데, attention은 encoder의 hs를 참고하는 모델이다"""
from _7_RNN_sentence import Encoder, Seq2seq, PeekySeq2seq
from time_layers import *

class AttentionEncoder(Encoder):
    def forward(self, xs):
        xs=self.embed.forward(xs)
        hs=self.lstm.forward(xs)
        return hs#hs전체를 사용하기에 마지막 hidden_state를 가져오는 부분을 제외한다.

    def backward(self, dhs):
        dout=self.lstm.backward(dhs)
        dout=self.embed.backward(dout)
        return dout

class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn
        
        embed_W=(rn(V,D)/100).astype('f')
        lstm_Wx=(rn(D,4*H)/np.sqrt(D)).astype('f')
        lstm_Wh=(rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b=np.zeros(4*H).astype('f')
        affine_W=(rn(2*H,V)/np.sqrt(2*H)).astype('f')
        affine_b=np.zeros(V).astype('f')

        self.embed=TimeEmbedding(embed_W)
        self.lstm=TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention=TimeAttention()#attention계층 추가
        self.affine=TimeAffine(affine_W, affine_b)
        layers=[self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads=[], []
        for layer in layers:
            self.params+=layer.params
            self.grads+=layer.grads

    def forward(self, xs, enc_hs):#입력 단어, 인코더 은닉상태전부
        h=enc_hs[:,-1]
        self.lstm.set_state(h)

        out=self.embed.forward(xs)
        dec_hs=self.lstm.forward(out)
        c=self.attention.forward(enc_hs, dec_hs)#인코더가,디코더입력이 반영된 h
        out=np.concatenate((c, dec_hs), axis=2)#ds와 결합
        score=self.affine.forward(out)

        return score#각 단어의 확률이 담긴 score(말그대로)

    def backward(self, dscore):
        dout=self.affine.backward(dscore)#affine'
        N,T,H2=dout.shape#현재 lstm h와 attention h가 concat된 상태
        H=H2//2#concat분할 attention의 h크기.

        dc, ddec_hs0=dout[:, :, :H], dout[:, :, H:]#어텐션 쪽과 디코더 은닉상태쪽값 분리
        denc_hs, ddec_hs1=self.attention.backward(dc)#어텐션 미분, 인코더hs미분값과 lstm미분값 get
        ddec_hs=ddec_hs0+ddec_hs1#초기에 LSTM에서 h가 분기되러 하나는 attention, 하나는 affine으로 들어갔기에 sum을 통한 미분
        dout=self.lstm.backward(ddec_hs)
        dh=self.lstm.dh#lstm backwarding결과로 저장된 lstm dh get
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):#enc_hs를 고려하여 start_id부터 sample_size만큼 sentence generate
        sampled=[]
        sample_id=start_id
        h=enc_hs[:, -1]#마지막 timestep의 h를
        self.lstm.set_state(h)#lstm의 초기 state로 설정..?

        for _ in range(sample_size):
            x=np.array([sample_id]).reshape((1,1))#sampled_id를 ndarray로

            out=self.embed.forward(x)
            dec_hs=self.lstm.forward(out)#얘와
            c=self.attention.forward(enc_hs, dec_hs)#얘를
            out=np.concatenate((c, dec_hs), axis=2)#concate해서 affine입력으로
            score=self.affine.forward(out)

            sample_id=np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled
"""Attention Encoder와 Attention Decoder을 연결해서 seq2seq를 만든다. 직관적으로 encoder의 forwarding을 decoder의 인자로 넘기면 된다."""
class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args=vocab_size, wordvec_size, hidden_size
        self.encoder=AttentionEncoder(*args)
        self.decoder=AttentionDecoder(*args)
        self.softmax=TimeSoftmaxWithLoss()

        self.params=self.encoder.params+self.decoder.params
        self.grads=self.encoder.grads+self.decoder.grads

#test
from dataset import sequence
from optimizers import Adam
from trainer import Trainer
from utils import eval_seq2seq
from _7_RNN_sentence import Seq2seq

(x_train, t_train), (x_test, t_test)=sequence.load_data('date.txt')
char_to_id, id_to_char=sequence.get_vocab()

x_train, x_test=x_train[:, ::-1], x_test[:, ::-1]

vocab_size=len(char_to_id)
wordvec_size=16
hidden_size=256
batch_size=128
max_epoch=10
max_grad=5.0#gradient cliping

model=AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)#현재 peeky seq2seq모델로 바꿔둠.
optimizer=Adam()
trainer=Trainer(model, optimizer)

acc_list=[]
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)#각 epoch마다 별도처리

    correct_num=0
    for i in range(len(x_test)):#테스트데이터 eval로 모델 향상 가시화
        question, correct=x_test[[i]], t_test[[i]]
        verbose=i<10
        correct_num+=eval_seq2seq(model, question, correct, id_to_char, verbose, is_reverse=True)

    acc=float(correct_num)/len(x_test)
    acc_list.append(acc)
    print('val acc: ', acc*100)
model.save_params()
#향상시키고 싶다면 reverse하는것도 나쁘지 않을듯 근데 이미 epoch2에서 99.42 val acc가 나와버리네. epoch3 시작 손실이 0.03..
