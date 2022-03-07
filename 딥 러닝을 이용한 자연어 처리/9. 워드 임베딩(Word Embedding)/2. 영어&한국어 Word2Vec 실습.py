    #[영어 Word2Vec 만들기]
"""import re
import urllib.request
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize

#1. 훈련 데이터 이해하기
#urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")
#전처리 필요_<content>...</content>만 가져오며, (Laughter) (Applause)같은 배경음 단어도 제거해야한다.

#2. 훈련 데이터 전처리하기
targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF-8')#XML open as UTF-8
target_text=etree.parse(targetXML)#parse XML

parse_text='\n'.join(target_text.xpath('//content/text()'))#get <content>...</content> by using xpath

content_text=re.sub(r'\([^)]*\)', '', parse_text)#remove (...) by regular expression. (으로 시작되어 )가 아닌(^)것들이 0개이상(*)있고 )로 마무리되는 pattern

sent_text=sent_tokenize(content_text)#현재 <content></content>로 감싸진 content중, 괄호 제거된 상태를 tokenization

normalized_text=[]
for string in sent_text:
    tokens=re.sub(r"[^a-z0-9]+", " ", string.lower())#a-z0-9가 아닌 문자열들을 " "으로 치환하며 lower(). (확실히 regex가 유용하긴 하구나..)
    normalized_text.append(tokens)
result=[word_tokenize(sentence) for sentence in normalized_text]

print('총 샘플의 개수: ', len(result),'\n상위 3개 샘플: ')
for line in result[:3]:
    print(line)

#3. Word2Vec 훈련시키기
from gensim.models import Word2Vec
from gensim.models import KeyedVectors#for loading saved Word2Vec model

model=Word2Vec(sentences=result, vector_size=100, window=5, min_count=5, workers=4, sg=0)#embedded_dim=100, limit_min_frequency=5, CBOW
#Hyper-parameter들로, size: embedding된 벡터의 차원, window=context window size, min_count: 단어 최소 빈도수 제한, sg=True는 Skip-gram, sg=False는 CBOW

model_result=model.wv.most_similar("man")
print("\nman가 비슷한 단어들: ", model_result)

#4. Word2Vec 모델 저장하고 로드하기
model.wv.save_word2vec_format('eng_w2v')
loaded_model=KeyedVectors.load_word2vec_format("eng_w2v")

model_result=loaded_model.most_similar("guy")
print("\n불러온 모델을 통한 computer와 비슷한 단어들:", model_result)
"""

    #[한국어 Word2Vec 만들기(네이버 영화 리뷰)]
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data=pd.read_table('ratings.txt')
print("상위 5개 행 출력: ", train_data[:5])

print('\n총 리뷰 개수: ', len(train_data))
print('결측값 유무 확인: ', train_data.isnull().values.any())#True. train_data에 isnull()을 모두 적용한 후, 그 values에게 any()연산을 적용

train_data=train_data.dropna(how='any')#drop null all 'any'
print('dropna후 결측값 유무 확인: ', train_data.isnull().values.any())
print('결측값 삭제 후의 리뷰개수:',len(train_data))

train_data['document']=train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")#한국어가 아니면 ""로 대체
print("한글 전처리후 상위 5개 출력: ", train_data[:5])

#불용어 제거
stopwords=['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '힌', '하다']
okt=Okt()

tokenized_data=[]
for sentence in train_data['document']:#tqdm은 for문의 진행표시바를 표시해주는 모듈이다.
    tokenized_sentence=okt.morphs(sentence, stem=True)#형태소 분리. stem은 형태소를 어간으로 바꿔준다(찍어야지->찍다)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]#불용어 필터링
    tokenized_data.append(stopwords_removed_sentence)
print('리뷰의 최대 길이: ', max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이: ', sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

#Word2Vec으로 학습
from gensim.models import Word2Vec

model=Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)
print("Size of embedding matrix of Word2Vec: ", model.wv.vectors.shape)

print("강호동과 비슷한 유사한 단어: ", model.wv.most_similar("강호동"))
print("군인과 유사한 단어: ", model.wv.most_similar("군인"))

    #[사전 훈련된 Word2Vec 임베딩(Pre-trained Word2Vec embedding) 소개]
