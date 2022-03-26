    #1. 기본 KeyBERT
import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

doc = """
드론 활용 범위도 점차 확대되고 있다. 최근에는 미세먼지 관리에 드론이 활용되고 있다.
서울시는 '미세먼지 계절관리제' 기간인 지난달부터 오는 3월까지 4개월간 드론에 측정장치를 달아 미세먼지 집중 관리를 실시하고 있다.
드론은 산업단지와 사업장 밀집지역을 날아다니며 미세먼지 배출 수치를 점검하고, 현장 모습을 영상으로 담는다.
영상을 통해 미세먼지 방지 시설을 제대로 가동하지 않는 업체와 무허가 시설에 대한 단속이 한층 수월해질 전망이다.
드론 활용에 가장 적극적인 소방청은 광범위하고 복합적인 재난 대응 차원에서 드론과 관련 전문인력 보강을 꾸준히 이어가고 있다.
지난해 말 기준 소방청이 보유한 드론은 총 304대, 드론 조종 자격증을 갖춘 소방대원의 경우 1,860명이다.
이 중 실기평가지도 자격증까지 갖춘 ‘드론 전문가’ 21명도 배치돼 있다.
소방청 관계자는 "소방드론은 재난현장에서 영상정보를 수집, 산악ㆍ수난 사고 시 인명수색·구조활동,
유독가스·폭발사고 시 대원안전 확보 등에 활용된다"며
"향후 화재진압, 인명구조 등에도 드론을 활용하기 위해 연구개발(R&D)을 하고 있다"고 말했다.
"""
#형태소 분석시 Okt이용, 명사만 분리
okt=Okt()#형태소 분석기
tokenized_doc=okt.pos(doc)#품사 태깅
tokenized_nouns=' '.join([word[0] for word in tokenized_doc if word[1]=='Noun'])#품사태깅 토큰중 Noun만 word추출.
print('품사 태깅 10개만 출력: ', tokenized_doc[:10])
print('명사 추출: ', tokenized_nouns)


#CountVectorizer이용, n-gram단위 단어 추출
n_gram_range=(2,3)#2개의 단어를 한 묶음으로 간주하는 bigram과 3개의 단어를 한 묶음으로 간주하는 trigram을 추출한다.

count=CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])#CountVectorizer Instantiation with fitting data
candidates=count.get_feature_names_out()#n-gram 항목들
print('trigram 개수: ', len(candidates))
print('trigram 5개만 출력: ', candidates[:5])

#SentenceTransformers이용, 다국어 SBERT로드
model=SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding=model.encode([doc])#문서 인코딩
candidate_embeddings=model.encode(candidates)#후보(n-gram단위) 인코딩

#상위 5개의 유사 키워드 출력
top_n=5
distances=cosine_similarity(doc_embedding, candidate_embeddings)#문서와 n-gram단위 후보들과의 코사인 유사도(거리) 저장
keywords=[candidates[index] for index in distances.argsort()[0][-top_n:]]#상위 5개 유사 후보 추출
print(keywords)


    #2. Max Sum Similarity: 후보끼리 안겹치면서 문서랑 유사하게
def max_sum_sim(doc_embedding, candidate_emebdding, words, top_n, nr_candidates):
    distances=cosine_similarity(doc_embedding, candidate_embeddings)#코사인 유사도는 1에 가까울 수록 유사도가 높다. 범위 [-1,1]
    distances_candidates=cosine_similarity(candidate_embeddings, candidate_embeddings)#아하 이걸 최소화하거 위에껄 최대화 하기 위해 두개를 구하는고만

    words_idx=list(distances.argsort()[0][-nr_candidates:])#nr_candidates만큼 doc-word유사도 높은 것중에서 top_n만큼 후보끼리 유사도 낮은걸 리턴 예정
    words_vals=[candidates[index] for index in words_idx]#실제 후보들 저장
    distances_candidates=distances_candidates[np.ix_(words_idx, words_idx)]#nr_candidates만큼 해당되는 후보들의 distances. 이제 이걸 바탕으로 후보간의 유사성 최소화 예정

    min_sim=np.inf
    candidate=None
    for combination in itertools.combinations(range(len(words_idx)), top_n):#0~len(words_index), top_n
        sim=sum([distances_candidates[i][j] for i in combination for j in combination if i!=j])#모든 경우들의 distance를 합하여 저장.
        #print([(i,j) for i in combination for j in combination if i!=j])
        if sim<min_sim:#만약 더 작다면
            candidate=combination#후보 등록을 한다
            min_sim=sim
    #무튼 가장 작은 유사도의 합을 갖는 후보가 남게됨.
    return [words_vals[idx] for idx in candidate]#그 후보의 인덱스를 word로 바꿔서 리스트 저장.
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10))
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30))


    #3. Maximal Marginal Relevance: 텍스트 요약에서 중복을 최소화, 다양성 극대화
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    word_doc_similarity=cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity=cosine_similarity(candidate_embeddings)

    keywords_idx=[np.argmax(word_doc_similarity)]#가장 높은 유사도 키워드의 인덱스
    candidates_idx=[i for i in range(len(words)) if i!=keywords_idx[0]]#keywords_idx제외 나머지 인덱스들

    for _ in range(top_n-1):#일단 하나 이미 argmax로 구해서 -1만큼 더 구함.
        candidate_similarities=word_doc_similarity[candidates_idx, :]#남은 후보들의 문서-키워드 유사도
        target_similarities=np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)#남은 후보들의 키워드-키워드 유사도 중 현재 포함된 keyword들의 유사도 최댓값

        mmr=(1-diversity)*candidate_similarities-diversity*target_similarities.reshape(-1,1)#남은 유사도들 - 키워드 단어끼리의 유사도.(만약 키워드 단어가 현재 너무 유사하다면 존나빼서 다른거) 이 모든것들의 예민함을 조절하는게 diversity!
        mmr_idx=candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
        
    return [words[idx] for idx in keywords_idx]
print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2))
print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7))
