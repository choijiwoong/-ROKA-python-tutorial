"""토픽들의 혼합으로 구성되는 토픽 모델링의 대표적인 알고리즘으로 확률분포에 기반하여 당너들을 생성한다고 가정하여 데이터가 주어지면 문서생성과정을 역추적한다.
 1. LDA 개요, 가정, 수행
LDA는 각 문서의 토픽 분포와 각 토픽 내의 단어분포를 추정한다. 입력으로 BoW행렬 DTM혹은 TF-IDF를 사용하여 순서와 무관하게 처리한다.
LDA는 문서의 작성자가 단어개수선정->확률분포기반 토픽혼합결정->확률적인 토픽선택 및 선택한 토픽의 단어출현확률분포에 기반하여 단어선정의 과정을 가정하여,
이를 역으로 추적하는 Reverse Engineering을 수행한다.
 LDA의 수행은 사용자가 토팩의 개수를 알려주면 모든 단어를 각각 하나의 토픽에 할당한다. 그 뒤 p(topic t | document d):문서 d의 단어들 중 토픽 t에 해당하는 단어들의 비율과
p(word w | topic t): 각 토픽들 t에서 해당 단어 w의 분포를 기반으로 재할당을 반복한다.(각 단어는 자신을 제외하고 나머지 단어들이 올바른 토픽에 할당되어있다고 가정)

 2. LSA와의 차이는 LSA는 DTM을 TruncatedSVD로 차원 축소하여 근접 단어들을 토픽으로 묶고, LDA는 확률추정으로 토픽을 추출한다."""
import pandas as pd#이전의 전처리 과정에서 tokenized_doc까지 동일
from sklearn.datasets import fetch_20newsgroups
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

dataset=fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents=dataset.data#[1]으로 접근하면 특수문자섞인 다수의 영어문장으로 구성. target_names로 접근하면 20개의 category확인가능.
print('\n샘플의 수: ', len(documents))

#(텍스트 전처리)
news_df=pd.DataFrame({'document': documents})#데이터를 dict 'document'에
news_df['clean_doc']=news_df['document'].str.replace("[^a-zA-Z]", " ")#cleaning후 데이터를 clean_doc에
news_df['clean_doc']=news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))#짧둥이 컷
news_df['clean_doc']=news_df['clean_doc'].apply(lambda x : x.lower())

stop_words=stopwords.words('english')
tokenized_doc=news_df['clean_doc'].apply(lambda x: x.split())#불용어 제거를 위한 토큰화
tokenized_doc=tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

#(정수 인코딩과 단어 집합 만들기)
from gensim import corpora

dictionary=corpora.Dictionary(tokenized_doc)#(정수인코딩, 빈도수) 꼴. 기존의 단어 확인은 인덱스 접근으로 확인이 가능하다.
corpus=[dictionary.doc2bow(text) for text in tokenized_doc]
print('dictionary의 길이: ', len(dictionary))

#(LDA 모델 훈련시키기)
import gensim

NUM_TOPICS=20
ldamodel=gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)#BoW행렬 입력(passed: 알고리즘이 corpus를 통과해야하는 횟수_확률추정과정 반복횟수같음)
topics=ldamodel.print_topics(num_words=4)#토픽별로 단어4개.
for topic in topics:
    print(topic)
#print(ldamodel.print_topics())#10개의 단어기준으로 출력

#(LDA 시각화하기)
import pyLDAvis.gensim_models

pyLDAvis.enable_notebook()
vis=pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)#좌측의 원은 20개의 토픽, 각 원과의 거리는 토픽들의 차이.(LDA모델의 출력 결과는 토픽의 번호가 1부터 시작하기에 0~19가 아닌 1~20을 가진다)

#(문서 별 토픽 분포 보기)
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic비율은: ', topic_list)
def make_topictable_per_doc(ldamodel, corpus):#보다 이쁜 출력
    topic_table=pd.DataFrame()

    for i, topic_list in enumerate(ldamodel[corpus]):
        doc=topic_list[0] if ldamodel.per_word_topics else topic_list#각 문서의 토픽들을 저장(문서벡터느낌)
        doc=sorted(doc, key=lambda x: (x[1]), reverse=True)#비중이 높은 순으로 정렬
        for j, (topic_num, prop_topic) in enumerate(doc):#해당 문서의 토픽들에서 토픽번호화 비중을
            if j==0:#비중이 가장 높은 토픽에 대하여(정렬되었기에)
                topic_table=topic_table.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_list]), ignore_index=True)#가장높은 번호&비중, 토픽리스트 저장.
            else:
                break
    return (topic_table)
topictable=make_topictable_per_doc(ldamodel, corpus)
topictable=topictable.reset_index()
topictable.columns=['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
print(topictable[:10])
