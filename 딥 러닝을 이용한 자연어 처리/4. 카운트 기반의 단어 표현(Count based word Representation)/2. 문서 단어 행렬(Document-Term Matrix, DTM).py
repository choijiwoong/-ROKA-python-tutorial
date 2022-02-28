"""[문서 단어 행렬]
서로 다른 문서들의 BoW를 결합한 표현방법이 문서단어행렬(DTM)이다. TDM이라고도 부르며, 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현한 것을 의미한다.
문서들을 서로 비교할 수있음에 의의를 갖으며, 필요에 따라서 형태소분석기로 단어 토큰화를 수행하며, 불용저를 제거하여 정제된 DTM을 얻을 수 있다.
 다만 DTM에도 한계는 존재하는데, 공간낭비에 대부분의 값이 0인 벡터(sparse vector, sparse matrix)는 높은 계산복잡도를 요구하는데 이를 Sparse Representation(희소문제)라고 부른다.
또한 단순한 빈도수기반 접근이기에 the같이 너무나도 당연하게 자주사용되는 것이 많다고 문서가 유사하다고 판단할 수 없다는 한계가 있다.
이를 해결하기 위해 TF-IDF는 불용어와 중요한 단어에 가중치를 주어 이들을 구분한다."""

"""[TF-IDF(Term Frequency-Inverse Document Frequency)]
단어의 빈도와 역 문서빈도(문서빈도에 특정식 이용)를 사용하여 단어들에 가중치를 주는 방법으로 DTM에 TF-IDF가중치를 부여하여 생성한다.
TF-IDE는 TF와 IDE의 곱을 의미하는데, 문서가 d, 단어가 t, 문서 총 개수가 n일때 TF, DF, IDF는 아래와 같다.
 (1) tf(d,t): 특정 문서 d에서의 특정단어 t의 등장 횟수
 (2) df(t): 특정 단어 t가 등장한 문서의 수
 (3) idf(d,t): df(t)에 반비례하는 수_log(n/(1+df(t))로 나타내는데, log는 IDF값이 기하급수적으로 커지는 것을 방지하는 역활이다.
               자주 쓰이지 안는 단어는 df에 반비례하는 idf에서 엄청난 가중치가 부여되기에 격차를 줄여준다. 또한 분모의 +1은 단순히 df가 없어(단어가 등장X) 분모 0이되는 것을  방지하기 위함이다.

TF-IDF는 자주등장하는 단어는 중요도가 낮게, 특정 문서에만 자주 등장하는 단어는 중요도가 높게 판단한다. DTM에서 tf를 얻은 뒤, ln(자연로그)를 사용하여 idf를 계산한 뒤 곱해준다.
"""

#파이썬으로 TF-IDF 직접 구현하기
import pandas as pd
from math import log

docs=[
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요'
]
vocab=list(set(w for doc in docs for w in doc.split()))#2개의 반복문
vocab.sort()

N=len(docs)

def tf(t, d):#d에서 t의 등장횟수
    return d.count(t)

def idf(t):#t가 등장한 문서의 수(df)의 반비례값
    #get df
    df=0
    for doc in docs:
        df+=t in doc#if false +0, else +1
        
    #calculater idf
    return log(N/(df+1))

def tfidf(t,d):
    return tf(t,d)*idf(t)#make tf-idf

#Get TF(DTM)
result=[]
for i in range(N):
    result.append([])#space for each document
    d=docs[i]
    for j in range(len(vocab)):
        t=vocab[j]
        result[-1].append(tf(t,d))#append tf for making DTM
tf_=pd.DataFrame(result, columns=vocab)#save by using dateframe
print("tf: ", tf_, end='\n\n')

#Get IDF
result=[]
for j in range(len(vocab)):
    t=vocab[j]#get each elememt of vocab
    result.append(idf(t))#append idf
idf_=pd.DataFrame(result, index=vocab, columns=['idf'])
print('idf: ', idf_, end='\n\n')

#Make TF-IDF Matrix
result=[]
for i in range(N):
    result.append([])
    d=docs[i]#docs
    for j in range(len(vocab)):
        t=vocab[j]#tern
        result[-1].append(tfidf(t,d))#call tfidf
tfidf_=pd.DataFrame(result, columns=vocab)
print('if-idf: ', tfidf_, end='\n\n')

"""위 방법에도 문제는 존재하는데, idf의 진수부에서 분모 +1로 분모와 분자가 같아져 1이 되면 idf가 0의 값을 갖고, 이는 가중치로서의 역활을 잃는 문제가 있다.
고로 이러한 문제가 보완된 사이킷런의 TF-IDF에서는 위의 식을 조정하여 사용하고 있다.
    [사이킷런을 이용한 DTM과 TF-IDF 실습]"""
from sklearn.feature_extraction.text import CountVectorizer

corpus=[
    'you know I want your love',#각각의 문서로 보자궁..
    'I like you',
    'what should I do',
]
vector=CountVectorizer()

print('BoW of corpus(by using sklearn): ', vector.fit_transform(corpus).toarray())#BoW(각각의 문서로 보자궁..) It's DTM!
print('vocabulary of corpus: ', vector.vocabulary_, end='\n\n')

#사이킷 런에서 TF-IDF를 자동계산해주는 TfidVectorizer을 제공한다(with 조정된 식)
from sklearn.feature_extraction.text import TfidfVectorizer

corpus=[
    'you know I want your love',#각각의 문서로 보자궁..
    'I like you',
    'what should I do',
]

tfidfv=TfidfVectorizer().fit(corpus)
print('tf-idf(by using TfidVectorizer): ', tfidfv.transform(corpus).toarray())
print('vocabulary of corpus: ', tfidfv.vocabulary_)
