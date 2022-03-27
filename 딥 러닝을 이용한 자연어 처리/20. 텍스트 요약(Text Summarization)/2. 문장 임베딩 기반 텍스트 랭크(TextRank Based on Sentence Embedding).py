"""앞선 seq2seq+attention기반 Abstractive Summarization 외에 텍스트 요약 다른 방법 중 하나인 Extractive Summarization을 해보자
    [1. 텍스트 랭크(TextRank)]
웹 페이지의 순위를 정하기 위해 사용되었던 알고리즘으로, 페이지 랭크를 기반으로 한 요약 알고리즘이다. 노드는 문장들이며, 간선 가중치는 유사도이다.

    [2. 사전 훈련된 임베딩(Pre-trained Embedding)]"""
import numpy as np
import gensim
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile

 #1. pre-trained GloVe download
urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip', filename='glove.6B.zip')
zf=zipfile.ZipFile('glove.6B.zip')
zf.extractall()
zf.close()

glove_dict=dict()
f=open('glove.6B.100d.txt', encoding='utf8')

for line in f:
    word_vector=line.split()
    word=word_vector[0]
    word_vector_arr=np.asarray(word_vector[1:], dtype='float32')
    glove_dict[word]=word_vector_arr#GloVe를 word: word_vector의 dict저장
f.close()
#glove_dict['cat']으로 임베딩 벡터 접근

 #2. pre-trained FastText download
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')#(왤케 오래걸리나 했는데 씌발 4기가네ㅋㅋ)
ft=fasttext.load_model('cc.en.300.bin')
#ft.get_word_vector('cat')으로 임베딩 벡터 접근

 #3. pre-trained Word2Vec download
urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", filename="GoogleNews-vectors-negative300.bin.gz")
word2vec_model=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
#word2vec_model['cat']으로 임베딩 벡터 접근


    #[3. 문장 임베딩(Sentence Embedding)]
#가장 쉬운 방법으로 단어 벡터들의 평균을 문장 임베딩으로 사용할 수도 있다. OOV문제를 해결하기 위해 OOV값도 추가해주자.
embedding_dim=100
zero_vector=np.zeros(embedding_dim)

def calculate_sentence_vector(sentence):
    return sum([glove_dict.get(word, zero_vector) for word in sentence])/len(sentence)#인자 2개 전하면 못찾을 경우 zero_vector리턴하는듯
#(test)
eng_sent=['I', 'am', 'a', 'student']
sentence_vector=calculate_sentence_vector(eng_sent)
print('영어 테스트 문장의 sentence vector 길이: ' ,len(sentence_vector))

kor_sent=['전', '좋은', '학생', '입니다']
sentence_vector=calculate_sentence_vector(kor_sent)
print('GloVe는 영어로 학습되었다: ', sentence_vector)


    #[4. 텍스트 랭크를 이용한 텍스트 요약]
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from urllib.request import urlretrieve
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx#원래 텍스트 랭크는 웹 페이지 순위를 구하기 위한 알고리즘.

stop_words=stopwords.words('english')

#텍스트 요약에 사용할 테니스 관련 기사 다운
urlretrieve("https://raw.githubusercontent.com/prateekjoshi565/textrank_text_summarization/master/tennis_articles_v4.csv", filename="tennis_articles_v4.csv")
data=pd.read_csv('tennis_articles_v4.csv')#article_id, article_text, source로 데이터 구성.

data=data[['article_text']]#article_text열만 추출.
data['sentences']=data['article_text'].apply(sent_tokenize)#nltk의 sent_tokenize사용 문장단위 토큰화하여 sentence열에 저장(sentences:문장단위 토큰화, article_text:원문)

#토큰화와 전처리 함수
def tokenization(sentences):
    return [word_tokenize(sentence) for sentene in sentences]

def preprocess_sentence(sentence):
    sentence=[re.sub(r'[^a-zA-Z\s]', '', word).lower() for word in sentence]#특수문자 제거, 소문자화
    return [word for word in sentence if word not in stop_words and word]#불용어 제거, 공백 제거
def preprocess_sentences(sentence):#위의 함수를 모든 문장에 대해 수행.
    return [preprocess_sentence(sentence) for sentence in sentences]

data['tokenized_sentences']=data['sentences'].apply(tokenization)#토큰화와 전처리 수행
data['tokenized_sentences']=data['tokenized_sentences'].apply(preprocess_sentences)#(tokenized_sentences:단어단위 토큰화, sentences:문장단위 토큰화, article_text:원문)

#문장벡터의 생성
embedding_dim=100#현재 사용할 GloVe의 차원은 100
zeros_vector=np.zeros(embedding_dim)

def calculate_sentence_vector(sentence):#단어 벡터의 평균을 반환
    if len(sentence)!=0:
        return sum([globe_dict.get(word, zero_vector) for word in sentence])/len(sentence)
    else:
        return zero_vector
def sentences_to_vectors(sentences):#위의 함수를 모든 문장에 실행
    return [calculate_sentence_vector(sentence) for sentence in sentences]

data['SentenceEmbedding']=data['tokenized_sentences'].apply(sentences_to_vectors)#(Sentence_Embedding:단어벡터평균, tokenized_sentences:단어단위 토큰화, sentences:문장단위 토큰화, article_text:원문)

#문장 벡터들의 코사인 유사도 행렬
def similarity_matrix(sentence_embedding):
    sim_mat=np.zeros([len(sentence_embedding), len(sentence_embedding)])
    for i in range(len(sentence_embedding)):
        for j in range(len(sentence_embedding)):#1차원으로 성형하여 cosine 유사도를 각각 계산. i번째 벡터값과 j번째 벡터값의 cosine유사도 lookup table반환
            sim_mat[i][j]=cosine_similarity(sentence_embedding[i].reshape(1, embedding_dim), sentence_embedding[j].reshape(1, embedding_dim))[0,0]
    return sum_mat

data['SimMatrix']=data['SentenceEmbedding'].apply(similarity_matrix)#(SimMatrix: 코사인 유사도, Sentence_Embedding:단어벡터평균, tokenized_sentences:단어단위 토큰화, sentences:문장단위 토큰화, article_text:원문)

print('두번째 샘플의 문장 개수: ', len(data['tokenized_sentences'][1]))#1(2)
print('두번째 샘플의 문장 벡터가 모인 문장 행렬의 크기(shape): ', np.shpe(data['SentenceEmbedding'][1]))#(12,100)
print('두번째 샘플의 유사도 행렬의 크기(shape): ', data['SimMatrix'][1].shape)#(12,12)

def draw_graphs(sim_matrix):#가시화 도구. 유사도 행렬을 그래프로.(nx 모듈 이용)
    nx.graph=nx.from_numpy_array(sim_matrix)
    plt.figure(figsize=(10,10))
    pos=nx.spring_layout(nx_graph)
    nx.draw(nx_graph, with_labels=True, font_weight='bold')
    nx.draw_networks_edge_labels(nx_graph, pos, font_color='red')
    plt.show()
draw_graphs(data['SimMatrix'][1])#두번째 샘플의 유사도행렬 그래프. 총 12문장이었기에 12개의 노드 존재.

def calculate_score(sim_matrix):
    nx_graph=nx.from_numpy_array(sim_matrix)#nx_graph를 ndarray에서 가져옴
    scores=nx.pagerank(nx_graph)#pagerank score계산
    return score
data['score']=data['SimMatrix'].apply(calculate_score)#(score: 페이지랭크점수, SimMatrix: 코사인 유사도, Sentence_Embedding:단어벡터평균, tokenized_sentences:단어단위 토큰화, sentences:문장단위 토큰화, article_text:원문)

def ranked_sentences(sentences, scores, n=3):#점수가 높은 상위 n개를 요약문으로 삼을 것이다.
    top_scores=sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_n_sentences=[sentence for score, sentence in top_scores[:n]]

    return " ".join(top_n_sentences)
data['summary']=data.apply(lambda x: ranked_sentences(x.sentences, x.score), axis=1)#(summary: 요약문, score: 페이지랭크점수, SimMatrix: 코사인 유사도, Sentence_Embedding:단어벡터평균, tokenized_sentences:단어단위 토큰화, sentences:문장단위 토큰화, article_text:원문)

for i in range(0, len(data)):
    print(i+1, '번 문서')
    print('원문: ', data.loc[i].article_text)
    print('')
    print('요약: ', data.loc[i].summary)
    print('')
