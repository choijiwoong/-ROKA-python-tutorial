""" 앞선 강의내용처럼 임베딩이 잘 되었다는 가정하에 단어 벡터들의 평균만으로 텍스트 분류를 잘 수행할 수 있다. 고로 워드 임베딩이 중요하다.
IMDB영화 리뷰 데이터는 label(sementic classification)을 이용한 데이터로, keras에서 바로 다운이가능하다."""
    #[1. 데이터 로드와 전처리]
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=20000#imdb.data_load()로 IMDB 리뷰 데이터를 다운할 건데, num_words 인자를 위한 변수로 등장 빈도 순위로 몇 번째에 해당하는 단어까지 사용할 것인지를 지정한다.
(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=vocab_size)#vocab_size안쪽 순위 빈도단어만 다운.
print('훈련용 리뷰 개수: ', len(X_train))
print('테스트용 리뷰 개수: ', len(X_test))
print('(test)훈련 데이터의 첫번째 샘플: ', X_train[0])#이미 integer encoding 전처리가 진행되어있는 것을 확인할 수 있다!
print('(test)훈련 데이터의 첫번째 샘플의 레이블: ', y_train[0])#1이면 positive!

#평균길이 check for padding
print('훈련용 리뷰의 평균 길이:', np.mean(list(map(len, X_train)), dtype=int))#238
print('테스트용 리뷰의 평균 길이:', np.mean(list(map(len, X_test)), dtype=int))#230

#padding
max_len=400#훈련용 리뷰와 테스트용 리뷰의 평균길이를 고려, 400으로 패딩길이 지정.

X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)
print('X_train의 크기(shape): ', X_train.shape)
print('X_test의 크기(shape): ', X_test.shape)

    #[2. 모델 설계하기]_내가 공부하기론 풀링은 CNN에서 대표값 설정하는 방법인데..?
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=64

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))#Embedding layer을 통과

model.add(GlobalAveragePooling1D())#1차원 단위에서 Pooling함수로 average구함.
model.add(Dense(1, activation='sigmoid'))#그 average로 binary_classification

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)#ㅋㅋㅋㄹㅇ overfitting, underfitting막기위해 있으면 좋겠다 생각했던 tool이 이제야 나오네 뜬금없지만 인내심=4존나 웃기네ㅋㅋ
mc=ModelCheckpoint('embedding_average_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)#콜백함수로 Keras모델 혹은 모델 가중치를 특정 빈도로 저장하기 위한 것이다.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])#손실계산
model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=[es, mc], validation_split=0.2)#callbacks로 EarlyStopping & ModelCheckpoint

#check
loaded_model=load_model('embedding_average_model.h5')#ModelCheckpoint로 저장된 model을 로드.
print('\n테스트 정확도: ', loaded_model.evaluate(X_test, y_test)[1])
