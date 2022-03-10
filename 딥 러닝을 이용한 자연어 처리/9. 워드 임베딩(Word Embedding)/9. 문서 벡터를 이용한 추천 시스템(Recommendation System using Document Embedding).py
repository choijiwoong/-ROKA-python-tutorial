""" 문서들을 고정된 길이의 벡터로 변환한다면 문서들을 비교할 수 있는데, 이러한 패키지로 Doc2Vec이나 Sent2Vec등이 있다.
혹은 문서의 단어 벡터들의 평균을 문서벡터로서 사용이 가능한데, 아래의 추천 시스템을 구현하여 학습해보자."""
 #[1. 데이터 로드]
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

#이미 따로 다운받아둠! https://drive.google.com/file/d/15Q7DZ7xrJsI2Hji-WbkU9j1mwnODBd5A/view?usp=sharing
#urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/data.csv", filename="data.csv")
df=pd.read_csv('data.csv')
print('전체 문서의 수:', len(df))
print('(test)상위 5개의 행 출력: \n', df[:5], '\n')

#cleaning tool
def _removeNonAscii(s):#s안의 i의 정수비교를 이용해서 아스키코드만 반환
    return "".join(i for i in s if ord(i)<128)#아스키 코드는 총 128개이다!

def make_lower_case(text):#소문자화
    return text.lower()

def remove_stop_words(text):#english의 stopword이용, cleaning
    text=text.split()
    stops=set(stopwords.words("english"))
    text=[w for w in text if not w in stops]
    text=" ".join(text)
    return text

def remove_html(text):
    html_pattern=re.compile('<.*?>')#<1글자 0회이상이 0회 또는 1회반복>
    return html_pattern.sub(r'', text)#sub(pattern, new_text, text)_text에서 pattern해당을 new_text로 대체하라.

def remove_punctuation(text):#구두점 제거
    tokenizer=RegexpTokenizer(r'[a-zA-Z]+')#regex를 사용한 tokenizer로, 영어단어만 regex 추출하여 토큰화!
    text=tokenizer.tokenize(text)
    text=" ".join(text)
    return text
#cleaning
df['cleaned']=df['Desc'].apply(_removeNonAscii)#Desc가 본문임. 이를 처리해서 cleaned 열에 저장하고, 그 열에 나머지 정제 저장.
df['cleaned']=df.cleaned.apply(make_lower_case)
df['cleaned']=df.cleaned.apply(remove_stop_words)
df['cleaned']=df.cleaned.apply(remove_punctuation)
df['cleaned']=df.cleaned.apply(remove_html)
print("(test)정제 전 상위 5개의 행 출력(Desc): \n", df['Desc'][:5])
print("(test)정제 후 상위 5개의 행 출력(cleaned): \n", df['cleaned'][:5], '\n')

#전처리 과정 중 불용어 제거하고..하며 아예 빈값이 된 행의 존재 가능성 고려, 빈행을 nan으로 변경 후에 제거.
df['cleaned'].replace('', np.nan, inplace=True)
df=df[df['cleaned'].notna()]#DataFrame(pd)의 notna()는 요소가 NA값이 아닌 지 여부를 나타내는 부울 값 마스크이다. 인덱스면 true, false의 행렬을 반환하는데
#이게 index로 사용되면 false로 된 열이 사라지나보다. 즉, 인덱스 참조할 행을 결정하는 느낌인가보다. 무튼 bool행렬을 인덱스로 받았더니 false행이 사라졌다.
print('nan변환 후 제거뒤 전체 문서의 수: ', len(df))#1개 줄음

#훈련시키기 위해 리스트로 변환한다.
corpus=[]
for words in df['cleaned']:
    corpus.append(words.split())

 #[2. 사전 훈련된 워드 임베딩 사용하기]
print('사전훈련된 Word2Vec가져오는중...')
urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", filename="GoogleNews-vectors-negative300.bin.gz")
print('사전훈련된 Word2Vec로딩완료')
word2vec_model=Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)#Word2Vec인스턴스화(argument size->vector_size Gensim 3.8->Gensim 4.0)
word2vec_model.build_vocab(corpus)#corpus등록??!!! 오류ㅠㅠㅠㅠ버전문제ㅠㅠㅠ왜 gensim github migration에 intersect에 관한 내용이 없는거야ㅠㅠ
word2vec_model.intersect_word2vec_format('GoogleNews-vectoers-negative300.bin.gz', lockf=1.0, binary=True)
word2vec_model.train(corpus, total_examples=word2vec_model.corpus_count, epochs=15)

 #[3. 단어 벡터의 평균 구하기]
def get_document_vectors(document_list):#문서들 안에 있는 문서의 평균벡터들을 리스트로 저장하는 document_embedding_list반환(현재 corpus가 문서들이라고 가정한다.)
    document_embedding_list=[]

    for line in document_list:#문서의 각 라인마다
        doc2vec=None
        count=0
        for word in line.split():#라인에 해당하는 모든 단어벡터를 doc2vec에 더하여 저장.
            if word in word2vec_model.wv.vocab:
                count+=1
                if doc2vec is None:
                    doc2vec=word2vec_model[word]
                else:
                    doc2vec=doc2vec+word2vec_model[word]#평균값을 구할 것이기에 embedding vector더함
        if doc2vec is not None:#라인의 벡터들의 평균값을
            doc2=doc2vec/count
            document_embedding_list.append(doc2vec)#document_embedding_list에 append
            
    return document_embedding_list
document_embedding_list=get_document_vectors(df['cleaned'])#df['cleaned']문서들의 평균벡터를 리스트로 받는다.
print('\n문서 벡터의 수: ', len(document_embedding_list),'\n')

 #[4. 추천 시스템 구현하기]
cosine_similarites=cosine_similarity(document_embedding_list, document_embedding_list)#문서에서 빈도수 기반이 아닌 벡터의 방향성으로 유사도 계산.
print('코사인 유사도 매트릭스 크기: ', cosine_similarities.shape)

def recommendations(title):#코사인 유사도를 이용, 줄거리가 유사한 5개의 책 찾아낸다.
    books=df[['title', 'image_link']]#title, image_link구조의 DataFrame생성
    
    indices=pd.Series(df.index, index=df['title']).drop_duplicates()#indices[title]=index꼴이 되게 만드는데, 중복제거
    idx=indices[title]#입력 title의 df에서의 index를 반환

    sim_scores=list(enumerate(cosine_similarities[idx]))#idx의 코사인 유사도를 iteratable 리스트로 저장.(모든 제목들의 코사인 유사도)
    sim_scores=sorted(sim_scores, key=lambda x:x[1], reverse=True)#코사인 유사도를 기준으로 [1]값을 이용하여 정렬
    sim_scores=sim_scores[1:6]#0은 자기니까 유사도 높은 1~5가져옴.

    book_indices=[i[0] for i in sim_scores]#그 항목들의 index0데이터를 book_indices로 가져옴
    recommend=books.iloc[book_indices].reset_index(drop=True)#인덱스를 처음부터 재배열
    fig=plt.figure(figsize=(20, 30))

    for index, row in recommend.iterrows():#row의 iterator
        response=requests.get(row['image_link'])#해당 recommand의 img르 가져와
        img=Image.open(BytesIO(response.content))#open
        fig.add_subplot(1, 5, index+1)
        plt.imshow(img)
        plt.title(row['title'])
print("'The Da Vinci Code'와 유사한 책: ", recommendations("the Da Vinci Code"))
#후...위에가 컴파일 되야 books의 구조를 출력해보고 구조를 이해할텐데 Gensim버전 충돌때매 확인을 못하니 반쪽짜리 공부네..
#그냥 오늘 알아갈 수 있는거는 문서의 단어벡터 평균으로 문서들을 비교할 수 있으며, 코사인 유사도를 통해 확인한다는 것 정도네.. 시간아까버라..
