import numpy as np

from utils import *

#(test of cos)
text='You say goodbye I say hello.'
corpus, word_to_id, id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus, vocab_size)

c0=C[word_to_id['you']]
c1=C[word_to_id['i']]
print(cos_similarity(c0, c1))

#(test)
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

