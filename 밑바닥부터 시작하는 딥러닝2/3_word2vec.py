from utils import preprocess, create_contexts_target, convert_one_hot
from layers import MatMul, SoftmaxWithLoss
from optimizers import Adam
from trainer import Trainer
import numpy as np

class SimpleCBOW:#step2
    def __init__(self, vocab_size, hidden_size):
        V, H=vocab_size, hidden_size

        W_in=0.01*np.random.randn(V,H).astype('f')#가중치 초기화
        W_out=0.01*np.random.rand(H,V).astype('f')

        self.in_layer0=MatMul(W_in)#계층 생성
        self.in_layer1=MatMul(W_in)
        self.out_layer=MatMul(W_out)
        self.loss_layer=SoftmaxWithLoss()

        layers=[self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads=[], []
        for layer in layers:
            self.params+=layer.params
            self.grads+=layer.grads

        self.word_vecs=W_in

    def forward(self, contexts, target):
        h0=self.in_layer0.forward(contexts[:,0])
        h1=self.in_layer1.forward(contexts[:,1])
        h=(h0+h1)*0.5#각 문맥의 vector의 평균을 입력으로
        score=self.out_layer.forward(h)#score계산
        loss=self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds=self.loss_layer.backward(dout)
        da=self.out_layer.backward(ds)
        da*=0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

#(test)
window_size=1
hidden_size=5
batch_size=3
max_epoch=1000

text='You say goodbye and I say hello.'
corpus, word_to_id, id_to_word=preprocess(text)#전처리를 통해 vocab받는다. tokenization with encoding

vocab_size=len(word_to_id)
contexts, target=create_contexts_target(corpus, window_size)#window_size로 context반영 데이터와 target생성 for CBOW
target=convert_one_hot(target, vocab_size)#one-hot encoding
contexts=convert_one_hot(contexts, vocab_size)

model=SimpleCBOW(vocab_size, hidden_size)
optimizer=Adam()
trainer=Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()
