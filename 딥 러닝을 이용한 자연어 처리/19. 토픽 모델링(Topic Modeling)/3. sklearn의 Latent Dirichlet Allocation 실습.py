#(뉴스 기사 제목 데이터에 대한 이해)
import pandas as pd
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/19.%20Topic%20Modeling%20(LDA%2C%20BERT-Based)/dataset/abcnews-date-text.csv", filename="abcnews-date-text.csv")
data=pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
print('뉴스 제목 개수: ', len(data))#1082168

#(텍스트 전처리)
text=data[['headline_text']]#뉴스 기사 제목만 추출

text['headline_text']=text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)#text의 각 데이터에 'headline_text'열 기준 토큰화(워닝 존나뜨네..괜찮은거가..)

stop_words=stopwords.words('english')#불용어 제거
text['headline_text']=text['headline_text'].apply(lambda x: [word for word in x if word not in (stop_words)])

text['headline_text']=text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])#표제어 추출(3인칭 단수를 1인칭, 동사를 현재형 등 길이를 줄이기 위해 사전기재된 뿌리 단어로 교체한다)

tokenized_doc=text['headline_text'].apply(lambda x: [word for word in x if len(word)>3])#짧둥이 컷

#(TF-IDF 행렬 만들기)
detokenized_doc=[]
for i in range(len(text)):
    t=' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
text['headline_text']=detokenized_doc

vectorizer=TfidfVectorizer(stop_words='english', max_features=1000)#최대 단어 1000개로 제한.
X=vectorizer.fit_transform(text['headline_text'])
print('TF-IDF행렬의 크기: ', X.shape)

#(Topic-Modeling)
lda_model=LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)#LDA인스턴스화. 토픽개수 10개
lda_top=lda_model.fit_transform(X)#데이터 피딩(detokenized data)

print(lda_model.components_)#토픽(10)xVocab_size(1000)
print(lda_model.components_.shape)

terms=vectorizer.get_feature_names()#vocab저장

def get_topics(components, feature_names, n=5):#lda_model의 components와 vocab전달
    for idx, topic in enumerate(components):#component의 각 토픽별로, 
        ㅜprint('Topic',idx+1,[(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n-1:-1]])#해당하는 상위 5개의 단어 출력(topic값에 대한 의문이 정수인코딩도 아니고 고민을 좀 해봤는데, 그냥 LDA의 연산 결과인듯)
get_topics(lda_model.components_, terms)
