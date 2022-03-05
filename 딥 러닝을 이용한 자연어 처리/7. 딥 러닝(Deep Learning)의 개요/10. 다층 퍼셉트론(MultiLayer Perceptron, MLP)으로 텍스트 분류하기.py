    #1. 다층 퍼셉트론(MultiLayer Perceptron, MLP)
#가장 기본적인 FFNN형태로, 실제 자연어 처리에는 RNN과 distributed representation을 사용한다.

    #2. 케라스의 texts_to_matrix() 이해하기
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts=['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']
tokenizer=Tokenizer()#Instantiation
tokenizer.fit_on_texts(texts)#tokenize
print("word_index: \n", tokenizer.word_index)#integer_encoding

print("texts_to_matrix(mode='count'): \n", tokenizer.texts_to_matrix(texts, mode='count'))#모드는 'binary', 'count', 'freq', 'tfidf'를 지원한다. 
#count의 경우 DTM(Document-Term Matrix)를 생성하는데, integer encoded vocabulary의 인덱스가 부여된다. 즉 [0,0,0,1]의 경우 vocab의 index3단어가 해당 문장에서 1번 등장했다는 것이다.
#DTM은 bag of words를 기반으로 하기에 단어 순서정보를 보존하지 않으며, 4개의 모든 모드에서 마찬가지다.
print("texts_to_matrix(mode='binary'): \n", tokenizer.texts_to_matrix(texts, mode='binary'))#존재하면 1
print("texts_to_matrix(mode='freq'): \n", tokenizer.texts_to_matrix(texts, mode='freq').round(2))#해당 문장의 단어중 특정단어가 몇%비율로 등장하는지(5개단어문장중 바나나 3번 0.6)
print("texts_to_matrix(mode='tfidf'): \n", tokenizer.texts_to_matrix(texts, mode='tfidf').round(2), '\n')#단어의 tf-idf 가중치. 모든문서자주등장시 감소, 특정문서 자주등장 증가(소수점2자리 반올림)

    #3. 20개 뉴스 그룹(Twenty Newsgroups)데이터에 대한 이해
import pandas as pd
from sklearn.datasets import fetch_20newsgroups#20개의 다른 주제를 가진 18,846개의 뉴스그룹 이메일 데이터
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

newsdata=fetch_20newsgroups(subset='train')#(훈련 데이터만 리턴)
print('newsdata의 키값: ', newsdata.keys())#총 6개의 속성 중 본문인 data와 주제가 기재된 숫자 레이블(target)이다.

print('훈련용 샘플의 개수:',format(len(newsdata.data)))
print('총 주제의 개수:', len(newsdata.target_names))
print(newsdata.target_names,'\n')

print('첫번째 샘플의 레이블:', newsdata.target[0])
print('첫번째 샘플의 레이블(7)이 의미하는 주제:', newsdata.target_names[7])
print('첫번째 샘플의 본문:', newsdata.data[0], '\n')

#데이터 프레임으로 data와 target을 정리하자
data=pd.DataFrame(newsdata.data, columns=['email'])
data['target']=pd.Series(newsdata.target)
print('5개 데이터: ', data[:5], '\n데이터 정보\n')
data.info()

print("\nnull값을 가진 sample이 있는지 확인:", data.isnull().values.any())
print("중복을 제외한 샘플의 수:",data['email'].nunique())
print('중복을 제외한 주제의 수:', data['target'].nunique())

#레이블 값의 분포 시각화
data['target'].value_counts().plot(kind='bar')
plt.show()

print("각 레이블이 몇개 있는지:\n", data.groupby('target').size().reset_index(name='count'))

#데이터프레임으로부터 데이터 불러오기
newsdata_test=fetch_20newsgroups(subset='test', shuffle=True)
train_email=data['email']
train_label=data['target']
test_email=newsdata_test.data
test_label=newsdata_test.target

#전처리 by keras의 tokenizer
vocab_size=10000
num_classes=20

def prepare_data(train_data, test_data, mode):#data를 tokenizer에 등록한 뒤 matrix와 vocab리턴
    tokenizer=Tokenizer(num_words=vocab_size)#num_words로 최대 단어 개수를 정의한다.(빈도수낮은거 컷)
    tokenizer.fit_on_texts(train_data)#train_data를 tokenizer에 장착
    X_train=tokenizer.texts_to_matrix(train_data, mode=mode)#samples*vocab_size
    X_test=tokenizer.texts_to_matrix(test_data, mode=mode)
    return X_train, X_test, tokenizer.index_word

X_train, X_test, index_to_word=prepare_data(train_email, test_email, 'binary')
y_train=to_categorical(train_label, num_classes)#one-hot
y_test=to_categorical(test_label, num_classes)
#그냥 모든 데이터 한번에 전처리 시키는거인듯

print('\n훈련 샘플 본문의 크기:', X_train.shape)
print('훈련 샘플 레이블의 크기:', y_train.shape)
print('테스트 샘플 본문의 크기:', X_test.shape)
print('테스트 샘플 레이블의 크기:', y_test.shape)

print('빈도수 상위 1번 단어:', index_to_word[1])#vocab_size가 빈도수 별로 컷됨. 그냥 잘 됬는지 확인
print('빈도수 상위 9999번 단어:', index_to_word[9999])

    #4. 다층 퍼셉트론(Multilayer Perceptron, MLP)을 사용하여 텍스트 분류하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def fit_and_evaluate(X_train, y_train, X_test, y_test):#input layer->hidden layer1->hidden layer2->output layer(DNN)
    model=Sequential()
    model.add(Dense(256, input_shape=(vocab_size,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=2, validation_split=0.1)#train과 validation용 data를 분리.
    score=model.evaluate(X_test, y_test, batch_size=128, verbose=0)#test데이터로 모델 테스트
    return score[1]#테스트 평가값(score_정확도) 반환

modes=['binary', 'count', 'tfidf', 'freq']

for mode in modes:
    X_train, X_test, _=prepare_data(train_email, test_email, mode)#모드별로 texts_to_metrics 인자가 달라지기에
    score=fit_and_evaluate(X_train, y_train, X_test, y_test)
    print(mode+' 모드의 테스트 정확도: ', score)
#freq모드는 적절한 전처리 방법이 아니라는 것을 알 수 있다.
