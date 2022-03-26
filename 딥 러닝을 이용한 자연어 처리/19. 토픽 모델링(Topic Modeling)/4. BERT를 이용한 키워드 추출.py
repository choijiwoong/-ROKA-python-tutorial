    #1. 기본 KeyBERT
import numpy as np
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

doc = """
         Supervised learning is the machine learning task of 
         learning a function that maps an input to an output based 
         on example input-output pairs.[1] It infers a function 
         from labeled training data consisting of a set of 
         training examples.[2] In supervised learning, each 
         example is a pair consisting of an input object 
         (typically a vector) and a desired output value (also 
         called the supervisory signal). A supervised learning 
         algorithm analyzes the training data and produces an 
         inferred function, which can be used for mapping new 
         examples. An optimal scenario will allow for the algorithm 
         to correctly determine the class labels for unseen 
         instances. This requires the learning algorithm to  
         generalize from the training data to unseen situations 
         in a 'reasonable' way (see inductive bias).
      """

n_gram_range=(3,3)#CountVectorizer에 인자로 n_gram_range를 사용하면 쉽게 n-gram추출가능! 현재의 경우 바로 trigram추출
stop_words='english'#n-gram을 사용하는 이유는 문서와 관련된 키워드를 도출하는 것이 이번 실습의 목적인데, 그 후보 키워드들을 의미하는 것! 후보들을 길이3씩 잘라 준비

count=CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])#count as trigram! data fitting
candidates=count.get_feature_names_out()#trigram의 keyword후보들을 저장
print('trigram개수: ', len(candidates))
print('trigram 5개출력: ', candidates[:5])

model=SentenceTransformer('distilbert-base-nli-mean-tokens')#Sentence Transformer 인스턴스화(BERT)
doc_embedding=model.encode([doc])#문서를 encoding
candidate_embeddings=model.encode(candidates)#candidates를 encoding. 이제 문서와 키워드의 유사도를 계산할 예정.

top_n=5
distances=cosine_similarity(doc_embedding, candidate_embeddings)#document와 candidates의 코사인 유사도 계산
keywords=[candidates[index] for index in distances.argsort()[0][-top_n:]]#상위 4개 키워드(상위의 index를 candidates에 넣어 실제 상위 4개 후보들 리스트를 만든다)
print(keywords)#문서를 잘 나타내는 상위 5개 키워드이기에 다 비슷비슷하다. 좀 더 다양하게 출력하려면 다양성의 미묘한 균형이 필요하며,
#이때 사용되는 알고리즘이 Max Sum Similarity와 Maximal Marginal Relevance이다.

    #2. Max Sum Similarity: 데이터 쌍 사이 거리가 최대화 되게하는 것으로 유사성을 최소화하며 문서와의 후보 유사성을 극대화한다(max니 없는값은 아니니까)
def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    distances=cosine_similarity(doc_embedding, candidate_embeddings)#문서,후보 코사인 유사도(거리)
    distances_candidates=cosine_similarity(candidate_embeddings, candidate_embeddings)#후보, 후보 코사인 유사도(거리)

    words_idx=list(distances.argsort()[0][-nr_candidates:])#인자로 입력된 만큼의 문서,후보 유사도 상위 추출.
    words_vals=[candidates[index] for index in words_idx]#index를 value로 매핑
    distances_candidates=distances_candidates[np.ix_(words_idx, words_idx)]#np.ix_는 슬라이싱?같은건데 np.ix([0,3],[2,3])의 경우 0,2 0,3 3,2 3,3데이터 가져옴. 지금의 경우 상위 후보index에 해당하는 후보거리 저장

    min_sim=np.inf
    candidate=None
    for combination in itertools.combinations(range(len(words_idx)), top_n):#이터레이터를 만든다
        sim=sum([distances_candidates[i][j] for i in combination for j in combination if i!=j])#word_index까지의 범위를 바꾸며 합의 최댓값을 추출한다.
        if sim<min_sim:
            candidate=combination
            min_sim=sim

    return [words_vals[idx] for idx in candidate]#최대 거리값을 같는 value를 리스트로 반환한다.
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10))
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=20))

    #3. Maximal Marginal Relevance: 문서와 유사한 키워드/키프레이즈 선택 뒤 반복적으로 새 후보를 선택
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    word_doc_similarity=cosine_similarity(candidate_embeddings, doc_embedding)#문서, 후보 코사인 유사도
    word_similarity=cosine_similarity(candidate_embeddings)#각 후보간의 코사인 유사도

    keywords_idx=[np.argmax(word_doc_similarity)]#문서, 후보 중 가장 높은 유사도를 가진 키워드의 인덱스(키워드가 비슷하지 않게한다하더라도 가장 큰 유사도는 그 자체로 기준키워드..나머지를 이거랑 다르게)
    candidates_idx=[i for i in range(len(words)) if i!=keywords_idx[0]]#최대 유사도를 제외한 나머지 후보 인덱스들.

    for _ in range(top_n-1):
        candidate_similarities=word_doc_similarity[candidates_idx, :]#나머지후보와 문서의 유사도
        target_similarities=np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)#후보와 키워드의 유사도 중 최댓값

        mmr=(1-diversity)*candidate_similarities-diversity*target_similarities.reshape(-1, 1)#후보*(1-다양성)-다양성*키워드유사도(다양성을 적용)
        mmr_idx=candidates_idx[np.argmax(mmr)]#다양성이 적용된 후보값중의 최댓값

        keywords_idx.append(mmr_idx)#keyword에 append(Maximal Marginal Relevance적용 후의 candidate max를 append
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]
print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.2))
print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.7))
