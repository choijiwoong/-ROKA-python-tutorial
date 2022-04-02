import numpy as np
from functions import softmax
from _6_RNN_with_gate import Rnnlm, BetterRnnlm

class Rnnlm(Rnnlm):#기존의 Rnnlm을 사용하여 start_id로 시작하는 문장 100사이즈로 생성
    def generate(self, start_id, skip_ids=None, sample_size=100):#시작할 입력단어, 패스할 단어들, 단어 몇개사용해서 문장만들건지
        word_ids=[start_id]

        x=start_id
        while len(word_ids)<sample_size:
            x=np.array(x).reshpae(1,1)
            score=self.predict(x)#Rnnlm predict
            p=softmax(score.flatten())#predict로 나온 score들 확률화 시킨다음에

            sampled=np.random.choice(len(p), size=1, p=p)#다음단어 추출의 확률분포로 사용
            if (skip_ids is None) or (sampled not in skip_ids):
                x=sampled
                word_ids.append(int(x))
                
        return word_ids
"""(test)
from dataset import ptb

corpus, word_to_id, id_to_word=ptb.load_data('train')
vocab_size=len(word_to_id)
corpus_size=len(corpus)

model=RnnlmGen()
#model.load_params(...)

start_word='you'
start_id=word_to_id[start_word]
skip_words=['N', '<unk>', '$']
skip_ids=[word_to_id[w] for w in skip_words]

word_ids=model.generate(start_id, skip_ids)
txt=' '.join([id_to_word[i] for i in word_ids])
txt=txt.replace(' <eos>', '.\n')
print(txt)"""
from time_layers import *
from base_model import BaseModel

class BetterRnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=650, hidden_size=650, dropout_ratio=0.5):
        V, D, H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn

        embed_W=(rn(V,D)/100).astype('f')
        lstm_Wx1=(rn(D,4*H)/np.sqrt(D)).astype('f')#세이비어
        lstm_Wh1=(rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b1=np.zeros(4*H).astype('f')
        lstm_Wx2=(rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b2=np.zeros(4*H).astype('f')
        affine_b=np.zeros(V).astype('f')#가중치 공유

        self.layers=[
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),#dropout추가
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),#LSTM계층 다중화
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)#weight share with embedding
        ]
        self.loss_layer=TimeSoftmaxWithLoss()
        self.lstm_layers=[self.layers[2], self.layerss[4]]
        self.drop_layers=[self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads=[], []
        for layer in self.layers:
            self.params+=layer.params
            self.grads+=layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg=train_flg
        for layer in self.layers:
            xs=layer.forward(xs)
        return xs#without loss function!

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

#seq2seq의 구현
class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn

        embed_W=(rn(V,D)/100).astype('f')#하나의 임베딩 층과 하나의 LSTM층을 사용한다.
        lstm_Wx=(rn(D,4*H)/np.sqrt(D)).astype('f')#오로지 hidden_state만 사용할 예정
        lstm_Wh=(rn(H,4*H)/np.sqrt(H)).astype('f')#참고로 4*H인 이유는 LSTM내부적으로 4개의 게이트에서 slice하여 사용하기에..
        lstm_b=np.zeros(4*H).astype('f')#크기: vocab_size를 같는 integer encoded단어가 embedding을 통하여 wordvec_size로 임베딩된다. 이 값이 LSTM의 입력으로 들어가는데, LSTM은 hidden_size만큼의 크기를 hidden_state로 출력한다. 다만 내부적으로 4개의 공통입력을 사용하기에 4배를 사용한다. 출력은 내부적 slicing으로 최종적으로 H가 된다.

        self.embed=TimeEmbedding(embed_W)
        self.lstm=TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params=self.embed.params+self.lstm.params#Embedding과 LSTM의 params저장
        self.grads=self.embed.grads+self.lstm.grads
        self.hs=None#게이모지 아 이따가 decoder로 전달할 최종 timestep의 hidden_state를 맴버변수로 저장

    def forward(self, xs):
        xs=self.embed.forward(xs)
        hs=self.lstm.forward(xs)
        self.hs=hs#통과 뒤의 LSTM hidden_state를 다음에 사용하기 위해 갱신
        return hs[:, -1, :]#마지막 은닉상태를 리턴하는 것으로 앞은 시퀀스의 단위개수, 뒤는 hidden_state이다. 중간은 timestep일거다..

    def backward(self, dh):
        dhs=np.zeros_like(self.hs)#dhs도 당연히 차원은 동일하니 self.hs shape로 초기화
        dhs[:, -1, :]=dh#크기 (시퀀스 단위, hidden_state)에서 (시퀀스단위, timestep, hidden_state)로 다시 원상복구

        dout=self.lstm.backward(dhs)
        dout=self.embed.backward(dout)
        return dout

class Decoder:#[참고로 인코더든 디코더든 내부적으로 TimeLSTM을 사용하고, 내부적으로 각 timestep끼리의 상호작용이 있는거라 여기서 사용하는 hidden_state들은 전부(최종) 출력값이다. 인 줄 알았는데 generate에 lstm계속하니 또 햇갈리쥬..]
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn

        embed_W=(rn(V,D)/100).astype('f')
        lstm_Wx=(rn(D,4*H)/np.sqrt(D)).astype('f')#xaiver
        lstm_Wh=(rn(H,4*H)/np.sqrt(H)).astype('f')
        lstm_b=np.zeros(4*H).astype('f')
        affine_W=(rn(H,V)/np.sqrt(H)).astype('f')#아핀계층이 추가. loss function도 있지만 학습과 훈련시 구성이 달라지기에 그 부분은 seq2seq class forward에서 구현예정이고 여기서도 forward와 generate의 구분으로 구현예정. 여기서 loss전까지만 구현
        affine_b=np.zeros(V).astype('f')

        self.embed=TimeEmbedding(embed_W)
        self.lstm=TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine=TimeAffine(affine_W, affine_b)
        #no loss func! it will be placed at seq2seq class

        self.params, self.grads=[], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params+=layer.params
            self.grads+=layer.grads

    def forward(self, xs, h):#loss계산용(train)
        self.lstm.set_state(h)#마지막 hidden_state로 lstm setting(초기엔 인코더값)

        out=self.embed.forward(xs)
        out=self.lstm.forward(out)
        score=self.affine.forward(out)
        return score#affine score를 리턴. not 확률

    def backward(self, dscore):
        dout=self.affine.backward(dscore)
        dout=self.lstm.backward(dout)
        dout=self.embed.backward(dout)#Decoder 모든 forward에 대하여 backwarding을 하며
        dh=self.lstm.dh#디코더의 입력으로 들어간 encoder의 hidden_state에 대하여 lstm의 dh값을 리턴
        return dh

    def generate(self, h, start_id, sample_size):#실제 생성용
        sampled=[]
        sample_id=start_id
        self.lstm.set_state(h)#input된 hidden_state로 lstm을 setting하는데, 초기엔 encoder's state가 들어올 예정

        for _ in range(sample_size):
            x=np.array(sample_id).reshape((1,1))
            out=self.embed.forward(x)
            out=self.lstm.forward(out)
            score=self.affine.forward(out)

            sample_id=np.argmax(score.flatten())#integer encoding값을 append
            sampled.append(int(sample_id))

        return sampled

class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H=vocab_size, wordvec_size, hidden_size
        self.encoder=Encoder(V, D, H)
        self.decoder=Decoder(V, D, H)
        self.softmax=TimeSoftmaxWithLoss()

        self.params=self.encoder.params+self.decoder.params
        self.grads=self.encoder.grads+self.decoder.grads

    def forward(self, xs ,ts):#softmaxwithloss!
        decoder_xs, decoder_ts=ts[:, :-1], ts[:, 1:]#ts에서 디코더에 사용할 full data를 가져와서 디코더 입력은 뒤에 하나 없는거, 디코더 레이블은 앞에 하나없는거로 데이터 split하여 만드는거네
        h=self.encoder.forward(xs)#xs는 인코더의 입력으로 들어가는 값이고 sequence겠네 사용된 encoder를 TimeLSTM이 구성하니.
        score=self.decoder.forward(decoder_xs, h)#xs즉 인코더 입력으로 들어간 후 인코더의 은닉상태를 decoder의 hidden_state로 시작하여 디코더 인풋xs전달했고
        loss=self.softmax.forward(score, decoder_ts)#나온값을 label데이터와 함께 손실계산. 여기서 softmax는 softmaxwithloss를 의미하기에 결과를 loss에 저장.
        return loss

    def backward(self, dout=1):#dsoftmaxwithloss!
        dout=self.softmax.backward(dout)
        dh=self.decoder.backward(dout)#태초 decoder에 들어간 encoder의 은닉상태에 대한 미분
        dout=self.encoder.backward(dh)
        return dout#softmaxwithloss->decoder->encoder

    def generate(self, xs, start_id, sample_size):#실제 생성!
        h=self.encoder.forward(xs)
        sampled=self.decoder.generate(h, start_id, sample_size)#인코더의 은닉상태를 디코더 은닉초기값으로!
        return sampled

#(evaluation!)
import matplotlib.pyplot as plt
from dataset import sequence
from optimizers import Adam
from trainer import Trainer
from utils import eval_seq2seq#구현 주석달며 공부하려다 디자인적인게 전부같아 그냥 배껴옴

(x_train, t_train), (x_test, t_test)=sequence.load_data('addition.txt')#get x,t train&test
x_train, x_test=x_train[:, ::-1], x_test[:, ::-1]#(1) REVERSE! 속도향상
char_to_id, id_to_char=sequence.get_vocab()#get vocab

vocab_size=len(char_to_id)
wordvec_size=16
hidden_size=128
batch_size=128
max_epoch=25
max_grad=5.0#clip_grad

model=Seq2seq(vocab_size, wordvec_size, hidden_size)#PeekySeq2seq!
optimizer=Adam()
trainer=Trainer(model, optimizer)

acc_list=[]
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
    
    correct_num=0
    for i in range(len(x_test)):#test x각 단어에 대하여
        question, correct=x_test[[i]], t_test[[i]]#해당 인덱스에 해당하는 decoder input과 label(output)을 추출하여
        verbose=i<10#10개 이하의 데이터에 대해서만 verbose활성화
        correct_num+=eval_seq2seq(model, question, correct, id_to_char, verbose)#eval_seq2seq는 정답일 시 1을 반환

    acc=float(correct_num)/len(x_test)
    acc_list.append(acc)
    print('검증 정확도: ', acc*100)
    
#학습속도를 향상시키는 방법으로는 (1) reverse to data를 통해 해당 label에 대응하는 단어와의 거리가 비교적 가까워지기에 기울기 전파를 보다 원활하게 할 수 있다.
#또한 (2) peeky즉 encoder의 정보를 포함하는 hidden_state를 모든 디코더의 lstm에 직접 전달하여 정확도를 높일 수 있다. 다만 매개변수가 커져서 연산속도가 느리다.

class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V,D,H=vocab_size, wordvec_size, hidden_size
        rn=np.random.randn

        embed_W=(rn(V,D)/100).astype('f')#정수인코딩, 인코더 h값 concate되서 H+D가 같이 들어가고 
        lstm_Wx=(rn(H+D, 4*H)/np.sqrt(H+D)).astype('f')#hidden_state가 concate되어 들어가기에 H+D, 출력하는건 4개의 hidden_state for inner calc
        lstm_Wh=(rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b=np.zeros(4*H).astype('f')
        affine_W=(rn(H+H, V)/np.sqrt(H+H)).astype('f')#LSTM에서 출력한 H과 또다시 인코더값 h가 concate되어 H+H가 들어가서 V가 나옴.(vocab_size)
        affine_b=np.zeros(V).astyle('f')

        self.embed=TimeEmbedding(embed_W)
        self.lstm=TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine=TimeAffine(affine_W, affine_b)

        self.params, self.grads=[], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params+=layer.params
            self.grads+=layer.grads
        self.cache=None

    def forward(self, xs, h):
        N,T=xs.shape#개수, timestep..?
        N,H=h.shape#개수, hidden_size

        self.lstm.set_state(h)

        out=self.embed.forward(xs)#out=개수, wordvec_size
        hs=np.repeat(h, T, axis=0).reshape(N,T,H)#각 timestep에 다 hs가 들어가야하니 repeat
        out=np.concatenate((hs, out), axis=2)#encoder의 state(정확히는 인자로 받은 그냥 h)를 embedding결과와 concatenate한다. (개수T, hidden_size+wordvec_size)

        out=self.lstm.forward(out)
        out=np.concatenate((hs, out),axis=2)#lstm결과와 인코더 state를 또 concatenate

        score=self.affine.forward(out)
        self.cache=H#역전파시 concat분리할때 사용 하는 크기정보
        return score

    def backward(self, dscore):
        H=self.cache#concat분리목적

        dout=self.affine.backward(dscore)
        dout, dhs0=dout[:, :, H:], dout[:, :, :H]#concatenate되었던거 분리. affine이전의 encoder's hs미분
        dout=self.lstm.backward(dout)
        dembed, dhs1=dout[:, :, H:], dout[:, :, :H]#lstm이전의 encoder's hs미분
        self.embed.backward(dembed)

        dhs=dhs0+dhs+1#하나의 hs가 분기되어 들어갔었으니 2배해주고
        dh=self.lstm.dh+np.sum(dhs, axis=1)#lstm의 dh와도 다 더해주어 태초 인코더에서 나온 상태의 미분값
        return dh

    def generate(self, h, start_id, sample_size):
        sampled=[]
        char_id=start_id
        self.lstm.set_state(h)

        H=h.shape[1]
        peeky_h=h.reshape(1, 1, H)
        for _ in range(sample_size):
            x=np.array([char_id]).reshape((1,1))
            out=self.embed.forward(x)

            out=np.concatenate((peeky_h, out), axis=2)
            out=self.lstm.forward(out)
            out=np.concatenate((peeky_h, out), axis=2)
            score=self.affine.forward(out)

            char_id=np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled

class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H=vocab_size, wordvec_size, hidden_size
        
        self.encoder=Encoder(V,D,H)
        self.decoder=PeekyDecoder(V,D,H)
        self.softmax=TimeSoftmaxWithLoss()

        self.params=self.encoder.params+self.decoder.params
        self.grads=self.encoder.grads+self.decoder.grads
