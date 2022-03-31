import numpy as np
import sys
import os

def preprocess(text):#(예시용)
    text=text.lower()
    text=text.replace('.', ' .')
    words=text.split(' ')

    word_to_id={}
    id_to_word={}
    for word in words:
        if word not in word_to_id:
            new_id=len(word_to_id)
            word_to_id[word]=new_id
            id_to_word[new_id]=word
    corpus=np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word#integer encoding, vocab반환

def create_co_matrix(corpus, vocab_size, window_size=1):#단어 벡터를 만드는 하나의 방법으로서, co-occurence matrix. corpus의 window단위 동시발생 횟수를 matrix화.
    corpus_size=len(corpus)
    co_matrix=np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):#각 단어별
        for i in range(1, window_size+1):#각 윈도우별
            left_idx=idx-i
            right_idx=idx+1

            if left_idx>=0:
                left_word_id=corpus[left_idx]
                co_matrix[word_id, left_word_id]+=1
                
            if right_idx<corpus_size:
                right_word_id=corpus[right_idx]
                co_matrix[word_id, right_word_id]+=1
    return co_matrix

def cos_similarity(x, y, eps=1e-8):#단어벡터의 유사도를 측정하기 위한 방법으로, 코사인 유사도는 벡터내적/각벡터의 L2노름 이다.
    nx=x/(np.sqrt(np.sum(x**2))+eps)#제곱의 합 sqrt(L2 norm)
    ny=y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx, ny)#product(내적)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):#(유사 단어의 랭킹) 검색어와 비슷한 단어를 유사도 순으로 출력한다.
    if query not in word_to_id:
        print("존재x")
        return

    print('\n[query] ',query)
    query_id=word_to_id[query]
    query_vec=word_matrix[query_id]

    vocab_size=len(id_to_word)#query에 대한 모든 단어들의 cosine_similarity계산
    similarity=np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i]=cos_similarity(word_matrix[i], query_vec)

    count=0
    for i in (-1*similarity).argsort():#해당하는 인덱스
        if id_to_word[i]==query:#자기자신 제외
            continue
        print(id_to_word[i], similarity[i])

        count+=1
        if count>=top:
            return

def ppmi(C, verbose=False, eps=1e-8):#co-occurence matrix는 고빈도 단어의 경우 단어와의 유사성(관련성)을 오인 할 수 있기에 개선된 통계 기반 기법인 점별 상호량.
    #Positive Pointwise Matual Information을 뜻하며 log2(P(x,y)/P(x)*P(y))를 의미한다.
    M=np.zeros_like(C, dtype=np.float32)
    N=np.sum(C)#인자로 들어온 동시발생행렬의 총합
    S=np.sum(C, axis=0)#단어개수
    total=C.shape[0]*C.shape[1]#전체 크기
    cnt=0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi=np.log2(C[i,j]*N/(S[j]*S[i])+eps)#모든 단어에 대하여 각각 pmi를 계산
            M[i,j]=max(0,pmi)#positive값을 위한 max(개선된 pmi)

            if verbose==True:
                cnt+=1
                if cnt%(total//100+1)==0:
                    print(100*cnt/total, '완료')
    return M

def create_contexts_target(corpus, window_size=1):#word2vec은 CBOW와 skip-gram이 있는데, CBOW의 경우 문맥을 반영하게 입력을 넣어준다. 그 여러 입력을 반환하는 함수
    target=corpus[window_size:-window_size]#target은 context사이에 낀 데이터를 말한다. label의 개념 of context
    contexts=[]

    for idx in range(window_size, len(corpus)-window_size):
        cs=[]
        for t in range(-window_size, window_size+1):#window사이즈만큼 append(window는 문맥 한번 볼 때 고려하는 단어 개수 like n-gram)
            if t==0:#context가 같아지는 경우는 제외 ex. contexts=[4,4]
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)#contexts에 넣고, 다음 탐색은 1칸 움직여서 탐색. 즉, window크기범위만큼씩 가능한 모든 범위를 append

    return np.array(contexts), np.array(target)#CBOW의 입력 데이터(문맥) 과 레이블 반환

def convert_one_hot(corpus, vocab_size):
    N=corpus.shape[0]#데이터의 양
    if corpus.ndim==1:#corpus 입력이 1차원 일 경우
        one_hot=np.zeros((N, vocab_size), dtype=np.int32)#데이터의 양만큼의 zeros matrix생성
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id]=1#각 단어의 순서별로 해당하는 one_hot matrix를 생성
    elif corpus.ndim==2:#corpus입력이 2차원 일 경우 단어별 해당하는 3차원 matrix반환
        C=corpus.shape[1]#데이터의 양
        one_hot=np.zeros((N, C, vocab_size), dtype=np.int32)#데이터의 양만큼 zeros matrix생성
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id]=1
    return one_hot#lookup table의 느낌임. one-hot이며, 기준은 단어별로 vocab_size만큼의 크기를 가지며 순서대로 0, 1,..., n인덱싱됨.

def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('calculating perplexity...')
    corpus_size=len(corpus)
    total_loss, loss_cnt=0, 0
    max_iters=(corpus_size-1)//(batch_size*time_size)#for stop condition
    jump=(corpus_size-1)//batch_size#for use batch

    for iters in range(max_iters):
        xs=np.zeros((batch_size, time_size), dtype=np.int32)#time_size고려 batch_size container2개
        ts=np.zeros((batch-size, time_size), dtype=np.int32)
        time_offset=iters*time_size#현재 위치
        offsets=[time_offset+(i*jump) for i in range(batch_size)]#현재위치&batch-size 고려 접근해야하는 index
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t]=corpus[(offset+t)%corpus_size]#배치1 예측
                ts[i, t]=corpus[(offset+t+1)%corpus_size]#배치1 정답

        try:
            loss=model.forward(xs, ts, train_flg=False)#해당 배치에 대한 연산 후 손실 리턴
        except TypeError:
            loss=model.forward(xs, ts)
        total_loss+=loss#총손실에 가산

        sys.stdout.write('\r%d / %d'%(iters, max_iters))#현황 출력
        sys.stdout.flush()
    print('')
    ppl=np.exp(total_loss/max_iters)#모든 loss를 iter횟수로 평균을 내어 exponential처리.
    return ppl#총 손실의 평균이기에 낮을 수록 모델의 성능이 좋다.

def eval_seq2seq(model, question, correct, id_to_char, verbose=False, is_reverse=False):#나머지는 나중에 나오면 해야거따
    correct=correct.flatten()
    start_id=correct[0]#머릿글자?
    correct=correct[1:]
    pass

def clip_grads(grads, max_norm):#뭐 이해는 됬는데 무슨 역활인질 모르겠네.. trainer클래스에서 사용됨 기울기의 정규화 느낌인듯.
    total_norm=0#이거 RNN의 고질적인 문제중 하나인 gradient explosion을 대비하기 위한 것! 문턱값(threshold)을 넘으면 특정 수치로 내리는데 여기선 max_norm!(인자)
    for grad in grads:
        total_norm+=np.sum(grad**2)
    total_norm=np.sqrt(total_norm)#각 기울기에 L2 Norm을 적용하고

    rate=max_norm/(total_norm+1e-6)#인자로 들어온 norm에 대하여 비율을 계산한 후
    if rate<1:#max가 total보다 작다면
        for grad in grads:
            grad*=rate#grad에 rate만큼을 곱해준다.

def analogy(a,b,c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    pass

def normalize(x):
    pass
