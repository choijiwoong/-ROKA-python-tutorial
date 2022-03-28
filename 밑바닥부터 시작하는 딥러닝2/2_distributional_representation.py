import numpy as np

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

"""(test of cos)
text='You say goodbye I say hello.'
corpus, word_to_id, id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus, vocab_size)

c0=C[word_to_id['you']]
c1=C[word_to_id['i']]
print(cos_similarity(c0, c1))
"""

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

"""(종합적인 사용 on PTB데이터셋)
from dataset import ptb
from sklearn.utils.extmath import randomized_svd#truncated SVD

window_size=2
wordvec_size=100

corpus, word_to_id, id_to_word=ptb.load_data('train')#데이터, vocab로드
vocab_size=len(word_to_id)

print('동시발생수 계산..')
C=create_co_matrix(corpus, vocab_size, window_size)#Co-occurence Matrix
print('pmi계산..')
W=ppmi(C, verbose=True)#PPMI계산(co-occurence의 유사도 저장)
print('SVD계산..')
U, S, V=randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)#특이값 분해

word_vecs=U[:, :wordvec_size]#truncated U를 단어 벡터로서 사용.

querys=['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
"""
