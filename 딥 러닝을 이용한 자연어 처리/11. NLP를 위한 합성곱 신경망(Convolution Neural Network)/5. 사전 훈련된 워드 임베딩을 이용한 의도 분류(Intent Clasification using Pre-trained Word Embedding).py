""" 의도 분류(Intent Classification)은 개체명 인식(Named Entity Recoginition)과 더불어 챗봇(Chatbot)의 중요 모듈로서 사용될 수 있다.
Pre-trained word embedding을 입력으로 intent classification을 해보자."""
 #1. 데이터 로드와 전처리
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

#Intent data
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/11.%201D%20CNN%20Text%20Classification/dataset/intent_train_data.csv", filename="intent_train_data.csv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/11.%201D%20CNN%20Text%20Classification/dataset/intent_test_data.csv", filename="intent_test_data.csv")
#load to DataFrame
train_data = pd.read_csv('intent_train_data.csv')
test_data = pd.read_csv('intent_test_data.csv'); print("(test) train_data:\n",train_data)

intent_train=train_data['intent'].tolist(); print('훈련용 문장의 수: ', len(intent_train))
label_train=train_data['label'].tolist(); print('훈련용 레이블의 수: ', len(label_train))
intent_test=test_data['intent'].tolist(); print('테스트용 문장의 수: ', len(intent_test))
label_test=test_data['label'].tolist(); print('테스트용 레이블의 수: ', len(label_test))
print('(test)훈련 데이터의 상위 5개 샘플: ', intent_train[:5])
print('(test)훈련 레이블의 상위 5개 샘플: ', label_train[:5])#데이터구조가 인덱스 2000씩을 기준으로 의도별로 구분해두었다.
print('(test)훈련 데이터 다음 의도: ', intent_train[2000:2002])
print('(test)훈련 레이블의 다음 의도: ', label_train[2000:2002])
print('(test)훈련 데이터 다음 의도: ', intent_train[4000:4002])
print('(test)훈련 레이블의 다음 의도: ', label_train[4000:4002],'\n')#등 index 2000씩을 기준으로 여러 의도별로 배치되어 있다.
train_data['label'].value_counts().plot(kind='bar'); plt.show() #시각화. 총 6개의 카테고리 존재(약 2000개 씩)

#6개의 의도 카테고리를 sklearn.preprocessing.LabelEncoder()이용 고유한 정수로 인코딩
idx_encode=preprocessing.LabelEncoder()
idx_encode.fit(label_train)#기준
label_train=idx_encode.transform(label_train)#label Integer encoding
label_test=idx_encode.transform(label_test)
label_idx=dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
print('레이블과 정수와의 맵핑 관계: ', label_idx,'\n')

print('intent_train[:5]:\n', intent_train[:5])#LabelEncoder를 통해 label데이터 integer encoding됨. 이제 intent데이터 처리할 시간
print('label_train[:5]:\n', label_train[:5])
print('intent_test[:5]:\n', intent_train[:5])
print('label_train[:5]:\n', label_train[:5],'\n')

#토큰화
tokenizer=Tokenizer()
tokenizer.fit_on_texts(intent_train)
sequences=tokenizer.texts_to_sequences(intent_train)
print("토큰화된 상위 5개 샘플:\n", sequences[:5])

word_index=tokenizer.word_index
vocab_size=len(word_index)+1
print('단어집합의 크기: ', vocab_size,'\n')

#패딩
print('문장의 최대 길이: ', max(len(l) for l in sequences))#35
print('문장의 평균 길이: ', sum(map(len, sequences))/len(sequences), '\n')#9.36
plt.hist([len(s) for s in sequences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len=25
intent_train=pad_sequences(sequences, maxlen=max_len)#padding
label_train=to_categorical(np.asarray(label_train))#원-핫 인코딩을 수행하는 이유는 다중 클래스 분류를 하기에 그 확률을 나타내기 위함이다.
print('훈련 데이터의 크기(shape): ', intent_train.shape)#11784, 25(paddind_size)
print('훈련 데이터 레이블의 크기(shape): ', label_train.shape)#11784, 6(label_category)
print('훈련 데이터의 첫번째 샘플: ', intent_train[0])#well integer encoded & padded
print('훈련 데이터의 첫번째 샘플의 레이블: ', label_train[0], '\n')#well one-hot vectorized

#데이터 분리
indices=np.arange(intent_train.shape[0])#이미 데이터가 인덱스 2000을 기준으로 종류별로 나뉘어 있기에 data_split전에 미리 섞어둬야한다.
np.random.shuffle(indices)
print('랜덤 시퀀스확인: ', indices)

intent_train=intent_train[indices]#이런문법은 처음보네..알아서 대입되는건가ㄷㄷ갓이썬
label_train=label_train[indices]

n_of_val=int(0.1*intent_train.shape[0])#validation data(10%)
print('검증 데이터의 개수: ', n_of_val,'\n')


X_train=intent_train[:-n_of_val]#앞쪽90%
y_train=label_train[:-n_of_val]
X_val=intent_train[-n_of_val:]#뒤쪽10%
y_val=label_train[-n_of_val:]

X_test=intent_test
y_test=label_test
print('훈련 데이터의 크기(shape): ', X_train.shape)#데이터의 호환을 위해 fitting전에 이렇게 shape비교해보는거 굉장히 좋은습관인듯..이전에 맞닥뜨린 오류들 shape로 바로 파악이 가능했던 문제들..
print('검증 데이터의 크기(shape): ', X_val.shape)
print('훈련 데이터 레이블의 크기(shape): ', y_train.shape)
print('검증 데이터 레이블의 크기(shape): ', y_val.shape)
print('테스트 데이터의 개수: ', len(X_test))
print('테스트 데이터 레이블의 개수: ', len(y_test))


 #2. 사전 훈련된 워드 임베딩 사용하기
embedding_dict=dict()#사전훈련된 vocab저장용
f=open(os.path.join('glove.6B.100d.txt'), encoding='utf-8')#이미 훈련된 GloVe
for line in f:#[word, 벡터들...]로 구성
    word_vector=line.split()
    word=word_vector[0]
    word_vector_arr=np.asarray(word_vector[1:], dtype='float32')
    embedding_dict[word]=word_vector_arr
f.close()
print('%s개의 Embedding vector정보를 저장했습니다.'%len(embedding_dict))
print("respectable의 GloVe벡터: ", embedding_dict['respectable'])
print("위의 벡터의 길이:", len(embedding_dict['respectable']), '\n')

embedding_dim=100#GloVe예시 하나 가져와 까보니 100임
embedding_matrix=np.zeros((vocab_size, embedding_dim))#Vocab단어전용 GloVe벡터 matrix제작예정.
print('임베딩 테이블의 크기(shape): ', np.shape(embedding_matrix))
for word, i in word_index.items():#vocab의 단어
    embedding_vector=embedding_dict.get(word)#대응되는 GloVe embedding vector를 가져와
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector#embedding_matrix에 차곡차곡 저장
print(embedding_matrix.shape)#9870, 100. 즉, vocab의 9870개 단어에 대해 embeddig_dim=100으로 저장된 벡터들의 matrix


 #3. 1D CNN을 이용한 의도 분류
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate

kernel_sizes=[2,3,5]
num_filters=512
dropout_ratio=0.5

model_input=Input(shape=(max_len,))#padded intent_train
output=Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(model_input)#GloVe weight를 이용할 것이기에 trainable=False

conv_blocks=[]
for size in kernel_sizes:
    conv=Conv1D(filters=num_filters, kernel_size=size, padding='valid', activation='relu', strides=1)(output)#3개의 kernel버전 conv에 각기 같은 output이 들어감. 추후 concatenate예정
    conv=GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

output=Concatenate()(conv_blocks) if len(conv_blocks)>1 else conv_blocks[0]#만약 길이가 미달이라면 불필요한 Concatenate없이 바로 conv_block[0]반환
output=Dropout(dropout_ratio)(output)
model_output=Dense(len(label_idx), activation='softmax')(output)
model=Model(model_input, model_output)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


history=model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))

epochs=range(1, len(history.history['acc'])+1)#acc 시각화
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title("model accuracy")
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

epochs=range(1, len(history.history['loss'])+1)#loss 시각화
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#evalutate
X_test=tokenizer.texts_to_sequences(X_test)
X_test=pad_sequences(X_test, maxlen=max_len)
y_predicted=model.predict(X_test)
y_predicted=y_predicted.argmax(axis=-1)#예측을 정수 시퀀스로변환
print('정확도(Accuracy): ', sum(y_predicted==y_test)/len(y_test))
