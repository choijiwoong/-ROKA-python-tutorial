#corpus로부터 각 개체의 유형을 인식하는 것으로, 유정이는 2018년에 골드만삭스에 입사했다. 라는 문장에서 사람, 조직, 시간에 대해 개체명 인식을 수행하는 것을 말한다.
    #[1. NLTK를 이용한 개체명 인식(Named Entity Recognition using NLTK)]_nltk에서 NER chunker을 지원하고 있다.
from nltk import word_tokenize, pos_tag, ne_chunk

sentence='James is working at Disney in London'
tokenized_sentence=pos_tag(word_tokenize(sentence))
print("tokenized_sentence with tag: ", tokenized_sentence)#품사태깅됨
ner_sentence=ne_chunk(tokenized_sentence)
print('tokenized_sentence with ne_chunk: ', ner_sentence,'\n\n')#개체명 인식됨

""" [2. 개체명 인식의 BIO표현 이해하기]
챗봇등의 주요 전처리 작업이며, 도메인 또는 목적에 특화되도록 개체명 인식을 정확하게 하려면 기존에 만들어진 대체명 인식기가 아닌
직접 목적에 맞는 데이터를 준비하여 모델을 만들어야만 한다.

 1. BIO표현
개체명인식의 보편적인 방법중 하나로, B: Begin(개체명시작), I: Inside(개체명내부), O: Outside(개체명아닌부분)으로 구분한다.
해리포터보러가자->BIIIOOOO로 나타내는 것이다. 이때 단순히 개체명인 부분을 표시하는것 뿐 아니라 어떤 종류인지도 태깅할 것이다.

 2. 개체명 인식 데이터 이해하기
CONLL2003은 개체명 인식을 위한 전통적인 영어 데이터셋으로, 데이터의 형식은 [단어] [품사태깅] [청크태깅] [개체명태깅]으로 구성된다.
개체명 태깅의 경우 LOC: location, ORG: organization, PER: person, MISC: miscellaneous(잡다한)을 의미한다.
BIO표현 방법을 사용하면 B-라는 태깅이 개체명 시작부분에 붙는다. 문장의 종료는 공란으로 표시한다. 연결되는 같은 속성은 B-PER, I-PER처럼 하나의 개체명으로 인식해준다.

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O

Peter NNP B-NP B-PER
Blackburn NNP I-NP I-PER"""
 #3. 데이터 전처리하기
import re
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

urllib.request.urlretrieve("https://raw.githubusercontent.com/jiacheng-ye/DocL-NER/master/data/conll2003/train.txt", filename="train.txt")
f=open('train.txt', 'r')
tagged_sentences=[]
sentence=[]#문장별로 모아서 append하기 위함 like buffer
for line in f:#(Peter NNP B-NP B-PER 꼴의 데이터를 전처리하는거임. 앞에 단어랑 뒤에 entity정보만 필요하다.)
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=='\n':
        if len(sentence)>0:
           tagged_sentences.append(sentence)
           sentence=[]#reset
        continue
    splits=line.split(' ')
    splits[-1]=re.sub(r'\n', '', splits[-1])#한 line끝에 에 \n이 있다면 없앤다.
    word=splits[0].lower()#lower하고
    sentence.append([word, splits[-1]])#단어와 Entity-info를 묶어 sentence에 append해버린다.
print('전체 샘플 개수: ', len(tagged_sentences))#14041
print('(test)첫번째 샘플: ', tagged_sentences[0],'\n')

#Sequences Labeling Task
sentences, ner_tags=[], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info=zip(*tagged_sentence)
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))
print('(test)첫번째 샘플의 문장: ', sentences[0])
print('(test)첫번째 샘플의 레이블: ', ner_tags[0])
print('(test)11번째 샘플의 문징: ', sentences[12])#길이가 제각각임을 알 수 있다!
print('(test)11번째 샘플의 레이블: ', ner_tags[12],'\n')

#데이터 길이분포 확인
print('샘플의 최대 길이: ', max([len(l) for l in sentences]))#113
print('샘플의 평균 길이: ', sum(map(len, sentences))/len(sentences),'\n')#14.50
plt.hist([len(sentence) for sentence in sentences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

#Integer encoding
vocab_size=4000#높은 빈도만 사용
src_tokenizer=Tokenizer(num_words=vocab_size, oov_token='OOV')#남은 값(빈도수<vocab_size)들은 OOV로 매핑
src_tokenizer.fit_on_texts(sentences)

tar_tokenizer=Tokenizer()#얜 작으니까..
tar_tokenizer.fit_on_texts(ner_tags)

tag_size=len(tar_tokenizer.word_index)+1
print('단어 집합의 크기: ', vocab_size)
print('개체명 태깅 정보 집합의 크기: ', tag_size)


X_train=src_tokenizer.texts_to_sequences(sentences)
y_train=tar_tokenizer.texts_to_sequences(ner_tags)
print('(test)Integer encoding후 첫번째 샘플의 문장: ', X_train[0])
print('(test)Integer encoding후 첫번째 샘플의 레이블: ', y_train[0], '\n')

#num_words의 사용으로 일부 단어가 'OOV'로 대체되었다. 디코딩 작업으로 확인해보자.
index_to_word=src_tokenizer.index_word
index_to_ner=tar_tokenizer.index_word

decoded=[]
for index in X_train[0]:#첫번째 샘플 ck
    decoded.append(index_to_word[index])
print('기존 문장: ', sentences[0])
print('빈도수가 낮은 단어가 OOV 처리된 문장: ', decoded, '\n')

#padding
max_len=70
X_train=pad_sequences(X_train, padding='post', maxlen=max_len)#sentence
y_train=pad_sequences(y_train, padding='post', maxlen=max_len)#tag

X_train, X_test, y_train, y_test=train_test_split(X_train, y_train, test_size=.2, random_state=777)

#one-hot encoding엥 순서 상관 있을텐데에.... 왜 품사태깅은 정수인코딩, 개체명 인식은 원핫인코딩..? 말 그대로 그 약간의 차이때문인가 품사는 순서 배치가 중요하고 개체명은 분류가 중요하니..?
y_train=to_categorical(y_train, num_classes=tag_size)
y_test=to_categorical(y_test, num_classes=tag_size)
print('훈련 샘플 문장의 크기: ', X_train.shape)
print('훈련 샘플 레이블의 크기: ', y_train.shape)
print('테스트 샘플 문장의 크기: ', X_test.shape)
print('테스트 샘플 레이블의 크기: ', y_test.shape, '\n')


 #4. 양방향 LSTM(Bi-directional LSTM)으로 개체명 인식기 만들기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from tensorflow.kera.optimziers import Adam

embedding_dim=128
hidden_units=128

model=Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True))#input_dim과 input_length 햇갈리지 말자!
