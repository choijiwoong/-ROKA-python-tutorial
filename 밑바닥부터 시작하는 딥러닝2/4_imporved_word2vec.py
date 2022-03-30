"""
1. 기존의 입력층의 MatMul계층을 Embedding으로 최적화
2. 은닉층 이후의 MatMul과 softmax계산을 Negative Sampling으로 해소. 이는 다중 분류를 이진 분류로 근사하는 것으로
context가 이거일때 답이 say인가요? 라는 질문에 yes or no로 대답하게 하는 것으로, say에 해당하는 열벡터만을 embedding_dot(내적)으로 추출 후 sigmoid로 확률화하면 된다.
즉, 중간에 확인하고자하는 단어의 입력을 추가로 넣는다."""
import numpy as np
from layers import Embedding, SigmoidWithLoss
import collections#for unigram sampler

class EmbeddingDot:#Embedding + MatMul
    def __init__(self, W):
        self.embed=Embedding(W)
        self.params=self.embed.params
        self.grads=self.embed.grads
        self.cache=None

    def forward(self, h, idx):#h가 context입력이며, idx가 negative sampling기법으로 binary-classification으로 만들기 위한 say.
        target_W=self.embed.forward(idx)#idx를 embedding
        out=np.sum(target_W*h, axis=1)#내적

        self.cache=(h, target_W)#for backward
        return out

    def backward(self, dout):
        h, target_W=self.cache
        dout=dout.reshape(dout.shape[0], 1)#열벡터화

        dtarget_W=dout*h#target(embedded idx)의 미분은 dot이기에 h를 곱하는데 dout이 이미 열벡터라 그냥 곱.(noe need transpose)
        self.embed.backward(dtarget_W)#say의 미분은 위에서 구한 target의 미분값을 embedding layer의 미분
        dh=dout*target_W#h즉 context입력은 embedded idx 즉 target과 dot된게 전부이기에 해당 target_W를 미분값에 곱함. 마찬가지로. dout은 이미 열벡터라 그냥 곱.
        return dh
    #이 미분이 헷갈린다면 그냥 모델에 대한 이해도가 낮은 걸로, negative sampling즉, multi-classification을 개선시키기 위해 binary_classification화하는걸로
    #원래 어떤 단어가 적절한지 구하는 다중분류를 특정 단어를 주고 맞냐 틀리냐를 구하는 이진분류로 바꿈. 이 어떤 단어가 forward인자 idx이며, 해당 단어를 embedding하고 context와 내적해야하기에
    #embedding을 거친 뒤에 matmul을 하는데 이를 한번에 귀찮으니 nagative sampling전용으로다가 쓰려고 Embedding+Dot product해서 EmbeddingDot클래스인거임.
    #그래서 이 EmbeddingDot의 입력은 특정 단어가 무엇인지와 문맥을 반영한 embedding input h인데, 고로 총 2개의 미분값을 backward에서 구함. 이게 dtarget_W와 dh
    #여기서 target은 embedding(특정단어) 즉 임베딩 후의 특정단어값을 부르는 이름일 뿐이고, 차례로 미분을 계산해보면 dtarget은 dot했기에 dout*h^T, 그 뒤 embed.backward로 최종적인 didx를 구할 수 있고,
    #dh는 target과 dot되었기에 dout에 target_W자체를 곱해주면 깔쌈하게 끝남.

"""위를 통해 다중분류를 이진분류로 바꾸었지만 우리는 정답인 say의 경우에서만 뭐 embeddingdot을 하고 label비교 어쩌구를 한다고 해서 위에 저지랄을 한거지
틀린 경우에 대해서도 학습을 시켜야하기에 negative sampling 즉 틀린 경우의 samping을 진행할거임. 즉 위에 뭐 get target함수쓰고 지랄났던거랑 비슷하게 틀린 예시를 뽑는 함수를 만들거임
일단 기본적인 툴로다가 unigram을 샘플링 해주는 함수를 직접 제작해보자."""
class UnigramSampler:#레알 찐 negative sampler on correct answer(target)
    def __init__(self, corpus, power, sample_size):#sampling에서 확률분포 이용할 시 0.75를 제곱하여 낮은 확률을 조금 크게 만들어서 낮은 확률의 단어가 추출될 가능성을 약간 상승시킨다.
        self.sample_size=sample_size
        self.vocab_size=None
        self.word_p=None

        counts=collections.Counter()
        for word_id in corpus:#빈도수 카운트 for make vocab
            counts[word_id]+=1

        vocab_size=len(counts)
        self.vocab_size=vocab_size

        self.word_p=np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i]=counts[i]#빈도수 큰 순서대로 word_p에 corpus integer encoded 단어저장.

        self.word_p=np.power(self.word_p, power)#오잉..정규화! 아하 지금 integer encoding된 것들이 드가있으니 최대한 정규화하여 random choice때 고루 뽑히게!
        self.word_p/=np.sum(self.word_p)

    def get_negative_sample(self, target):#target에 대한 틀린 context반환
        batch_size=target.shape[0]

        #if not GPU:
        negative_sample=np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p=self.word_p.copy()
            target_idx=target[i]
            p[target_idx]=0#뽑히지 않게
            p/=p.sum()#re normalize
            negative_sample[i, :]=np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)#recursive

        #negative_sample=np.choice(self.vocab_size, size=(batch_size, self.sample_size), replace=True, p=self.word_p)

        return negative_sample

class NegativeSamplingLoss:#모든 것이 합쳐졌다...EmbeddingDot도 UnigramSampler도..Loss func도..
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size=sample_size
        self.sampler=UnigramSampler(corpus, power, sample_size)#****1) sampler에 샘플 사이즈 전달. 
        self.loss_layers=[SigmoidWithLoss() for _ in range(sample_size+1)]#샘플마다 각각 loss func와(긍정예시 하나, 부정예시 sample_size개 해서 총 sample_size+1)
        self.embed_dot_layers=[EmbeddingDot(W) for _ in range(sample_size+1)]#embed가 필요하다! multi->binary의 숙명(긍정예시 하나, 부정예시 sample_size개 해서 총 sample_size+1)
        self.params, self.grads=[], []
        for layer in self.embed_dot_layers:#Emb의 매개변수.(아마 같이 학습할 거라 통일용일듯. 이 학습의 주인공)
            self.params+=layer.params
            self.grads+=layer.grads

    def forward(self, h, target):
        batch_size=target.shape[0]
        negative_samples=self.sampler.get_negative_sample(target)#negative sample을 계산. sample_size개의 negative_sample이 저장.

        #긍정 예시의 순전파
        score=self.embed_dot_layers[0].forward(h, target)#긍정에는 0번꺼 사용
        correct_label=np.ones(batch_size, dtype=np.int32)#positive 예시이기에 batch_size전부 1로.
        loss=self.loss_layers[0].forward(score, correct_label)#loss calc

        #부정 예시의 순전파
        negative_label=np.zeros(batch_size, dtype=np.int32)#negative 예시이기에 batch_size전부 0으로.
        for i in range(self.sample_size):
            negative_target=negative_samples[:, i]#해당 번째에 해당하는 embed_dot의 target.
            score=self.embed_dot_layers[1+i].forward(h, negative_target)#(i+1의 이유는 단순히 0이 positive용이라서..)target과 h의 embed_dot연산
            loss+=self.loss_layers[1+i].forward(score, negative_label)#손실에 ++

        return loss#긍정예측, 부정예측의 손실이 ++된 loss반환

    def backward(self, dout=1):
        dh=0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore=l0.backward(dout)#loss func의 미분
            dh+=l1.backward(dscore)#embed_dot의 미분, 그 값을 dh에 ++

        return dh

#1. 최종적인 CBOW 모델의 구현
class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):#window_size는 context를 반영하는 거의 개수
        V, H=vocab_size, hidden_size

        W_in=0.01*np.random.randn(V,H).astype('f')#Weight initialization
        W_out=0.01*np.random.randn(V,H).astype('f')

        self.in_layers=[]#계층 생성
        for i in range(2*window_size):#context크기만큼 embedding예정
            layer=Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss=NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)#EmbDot+Loss with Sampling

        layers=self.in_layers+[self.ns_loss]#(손실함수로 layers에 추가)
        self.params, self.grads=[], []#가중치와 기울기 한곳에 모으기
        for layer in layers:
            self.params+=layer.params
            self.grads+=layer.grads

        self.word_vecs=W_in#W_in은 word_vecs에 해당하니 따로 instance variable에 저장

    def forward(self, contexts, target):
        h=0#contexts들의 임베딩의 평균값을 구하기 위한 결과값 더하기목적의 버퍼..느낌
        for i, layer in enumerate(self.in_layers):#context크기만큼의 layer들에 대하여
            h+=layer.forward(contexts[:, i])#각각의 context들을 forward(embedding)
        h*=1/len(self.in_layers)#그 후 평균값을 낸 뒤
        loss=self.ns_loss.forward(h, target)#실제 negativesamplingloss with target단어
        return loss

    def backward(self, dout=1):
        dout=self.ns_loss.backward(dout)#Negative Sampling with Loss 의 모든 backward결과 저장.
        dout*=1/len(self.in_layers)#아까 얘랑 곱해졌었는데 지금은 dout이 궁금하지 davg?가 궁긍한게 아니라 바뀌어져서 dout과 곱.
        for layer in self.in_layers:
            layer.backward(dout)#emb에 대해서도 backward
        return None


#2. CBOW 모델의 학습
import pickle
from trainer import Trainer
from optimizers import Adam
from utils import create_contexts_target, most_similar
from dataset import ptb

window_size=5
hidden_size=100
batch_size=100
max_epoch=10

corpus, word_to_id, id_to_word=ptb.load_data('train')
vocab_size=len(word_to_id)

contexts, target=create_contexts_target(corpus, window_size)

model=CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer=Adam()
trainer=Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs=model.word_vecs#학습된 word2vec을 저장

params={}
params['word_vecs']=word_vecs.astype(np.float16)
params['word_to_id']=word_to_id
params['id_to_word']=id_to_word
plk_file='cbow_params.pkl'
with open(plk_file, 'wb') as f:
    pickle.dump(params, f, -1)

#3. CBOW 모델 평가
pkl_file='cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params=pickle.load(f)

    word_vecs=params['word_vecs']
    word_to_id=params['word_to_id']
    id_to_word=params['id_to_word']

querys=['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
