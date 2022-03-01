"""문장이나 문서의 유사도를 구하기 이전에 처리를 위해 이를 벡터로 Encoding하기에 이러한 수치화 방법(DTM, Word2Vec),
문서 단어들의 차이를 어떤 방법(유클리드 거리, 코사인 유사도)에 따라 달라진다.

    [코사인 유사도(Cosine Similarity)]
두 벡터 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도이다. 방향이 동일하면 1, 반대의 방향을 가지면 -1의 값을 가진다.(수직0)
"""
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))#코사인 유사도 공식. norm은 벡터의 크기, 길이를 의미한다. (자기 자신과의 내적에 의해 구해진다?)

doc1=np.array([0,1,1,1])#저는 사과 좋아요
doc2=np.array([1,0,1,1])#저는 바나나 좋아요
doc3=np.array([2,0,2,2])#저는 바나나 좋아요 저는 바나나 좋아요
#를 바나나 사과 저는 좋아요 순으로 토큰화했다고 가정한 행렬.

print('문서1과 문서2의 유사도: ', cos_sim(doc1, doc2))
print('문서1과 문서3의 유사도: ', cos_sim(doc1, doc3))
print('문서2와 문서3의 유사도: ', cos_sim(doc2, doc3), end='\n\n\n')#빈도수가 1씩 등장한 경우 코사인 유사도가 1이다. 즉, 비교하는 문서의 양(길이)가 달라서
#유클리드 거리 연산시 빈도율을 같지만 단순히 소스의 양이 많아 유사도가 다른 경우에 코사인 유사도를 사용하면 vector방향만 보지 scalar값은 무시되기에 해결책이 될 수 있다.


    #[유사도를 이용한 추천 시스템 구현하기]
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_csv('movies_metadata.csv', low_memory=False)#https://www.kaggle.com/rounakbanik/the-movies-dataset
print("상위 2개 데이터 of movies_metadata.csv", data.head(2))#overview에 코사인 유사도를 사용할 예정이다(줄거리)

data=data.head(20000)#상위 2만개의 샘플을 data에 저장
print('overview 열의 결측값의 수: ', data['overview'].isnull().sum())#overview가 null인 결측값의 수 계산

data['overview']=data['overview'].fillna('')#null(결측값)을 ''(empty value)로 대체_fill null all

tfidf=TfidfVectorizer(stop_words='english')#불용어
tfidf_matrix=tfidf.fit_transform(data['overview'])#td-idf matrix생성(각 단어에 가중치)
print('TF-IDF 행렬의 크기(tfidf_matrix.shape): ', tfidf_matrix.shape)#20000행, 47847열(단어)

cosine_sim=cosine_similarity(tfidf_matrix, tfidf_matrix)
print('코사인 유사도 연산 결과(cosine_similarity(tfidf_matrix, tdidf_matrix): ', cosine_sim.shape)#상호 유사도가 기록된 행렬. return ndarray(n_samples_X, n_samples_Y)

title_to_index=dict(zip(data['title'], data.index))#title을 index로 바꾸는 dictionary생성.(zip은 tuple화 시킨다)

def get_recommendations(title, cosine_sim=cosine_sim):#cosine_sim값을 기존에 tdidf_matrix를 이용하여 생성한 cosim_similarity로 설정한다.
    idx=title_to_index[title]#입력 title을 index로

    sim_scores=list(enumerate(cosine_sim[idx]))#index의 cosine_sim을 list로.
    sim_scores=sorted(sim_scores, key=lambda x:x[1], reverse=True)#내림차순 정렬

    sim_scores=sim_scores[1:11]#상위항목 10개 slicing. 0은 자기 자신일테니(동일한 영화)
    #(sim_scores는 index0에 본래 index, index1에 similarity를 저장하고 있다. print해보면 암).
    movie_indices=[idx[0] for idx in sim_scores]#sim_scores(유사도 높은 10개항목)의 index[0]을 가져온다.(본래 영화 위치(index))
    #index와 movie_indices가 헷갈려서 정리하면, sim_scores슬라이싱으로 그냥 상위 10개 항목을 가져온거고 그 sim_scores가 가지고 있는 영화정보중 index를 리턴을 위해 뽑음.
    return data['title'].iloc[movie_indices]#movid_indices(sim_score이 높은 영화의 index)로 data의 title이 행만 가져오기
print(get_recommendations('The Dark Knight Rises'))
