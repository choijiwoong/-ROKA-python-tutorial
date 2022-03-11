#로이터 뉴스는 keras에서 제공하며, 11,258개의 뉴스 기사가 46카테고리로 분류되는 뉴스 기사 데이터이다. 기본적으로 유용한 메소드들을
#데이터단에서 지원하는데, 대체로 pandas data와 유사하다.
    #[1. 로이터 뉴스 데이터에 대한 이해]
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters

(X_train, y_train), (X_test, y_test)=reuters.load_data(num_words=None, test_split=0.2)#빈도수 컷(num_words) 제한 없이 모든 데이터 가져오는데 테스트 0.2비율 split
print('훈련용 뉴스 기사개수:', len(X_train))#8982(8:2 as test_split argument)
print('테스트용 뉴스 기사개수:', len(X_test))#2246
num_classes=len(set(y_train))
print('카테고리: ', num_classes,'\n')#46

print('(test)첫번째 훈련용 뉴스 기사: ', X_train[0])#이미 keras.reuters상에서 tokenization, integer encoding이 끝난 상태. 이는 데이터의 등장 빈도를 의미하며 출력시 확인가능한 1이라는 token은 등장빈도가 1등인 단어라는 것이다(integer eocnding이 완료된 상태기에)
print('(test)첫번째 훈련용 뉴스 기사의 레이블: ', y_train[0],'\n')

#길이 분포를 확인
print('뉴스 기사의 최대 길이: ', max(len(sample) for sample in X_train))
print('뉴스 기사의 평균 길이: ', sum(map(len, X_train))/len(X_train))
plt.hist([len(sample) for sample in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()#평균적으로 100~200사이의 길이를 가진다.

#각 뉴스의 레이블 값(카테고리)의 분포 확인
fig, axe=plt.subplots(ncols=1)
fig.set_size_inches(12,5)
sns.countplot(y_train)
plt.show()#대부분 3,4 카테고리

#위의 seaborn을 이용한 분포에서 3,4편향을 확인하고 보다 정확한 레이블별 기사 개수 확인
unique_elements, counts_elements=np.unique(y_train, return_counts=True)#y_train에서 카운트정보를 포함하여 unique정보를 반환.
print('각 레이블에 대한 빈도수: ')
print(np.asarray((unique_elements, counts_elements)))#(카테고리 레이블과 해당하는 빈도수를 따로 배열로 출력. 별 중요하진 않음)
#3번이 3,159개, 4번이 1,949개. 이들이 의미하는게 어느 카테고리인지를 확인하기 위해 word_index 출력
word_to_index=reuters.get_word_index()
print('word_to_index: ', word_to_index,'\n')

index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value+3]=key#로이터 뉴스 데이터셋 규칙 상 저장된값에 +3을 해야 실제 매핑되는 정수이다.
print("(test)빈도수 상위 1번 단어: ", index_to_word[4])#index_to_word에서 0은 pad, 1은 문장의 시작을 알리는 sos, 2는 OOV토큰인 unk토큰으로 매핑되어있다.
print("(test)빈도수 상위 128등 단어: ", index_to_word[131],'\n')

for index, token in enumerate(('<pad>', '<sos>', '<unk>')):#첫번째 기사를 복원해본다.(index=[0,1,2], token=['<pad>', '<sos>', '<unk>']
    index_to_word[index]=token#로이터 뉴스규칙상 저장된 0~2번인덱스를 일반적인 문자열로 바꿔준다
print(' '.join([index_to_word[index] for index in X_train[0]]))#X_train[0]의 index에 대해 index_to_word를 통해 문장을 얻는다.


    #[2. LSTM으로 로이터 뉴스 분류하기]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

vocab_size=1000#빈도순위 상위1000개 단어만 사용
max_len=100#뉴스 기사의 길이를 100으로 패딩 예정

(X_train, y_train), (X_test, y_test)=reuters.load_data(num_words=vocab_size, test_split=0.2)
X_train=pad_sequences(X_train, maxlen=max_len)#padding
X_test=pad_sequences(X_test, maxlen=max_len)
y_train=to_categorical(y_train)#one-hot encoding
y_test=to_categorical(y_test)

embedding_dim=128#hyperparameters
hidden_units=128
num_classes=46

model=Sequential()#Embedding->LSTM(hidden_state & cell_state)->Dense
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(num_classes, activation='softmax'))

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)#EarlyStopping patience=4로 가동시키며
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)#best weight만 기록. 모드에 따라 save시 어떤 값을 기준으로 할지 결정한다. auto min max가 있다. val_loss로 할시 당연히 min이고, val_acc로 할시 max이다.

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history=model.fit(X_train, y_train, batch_size=128, epochs=30, callbacks=[es, mc], validation_data=(X_test, y_test))#검증을 test로,

loaded_model=load_model('best_model.h5')
print('\n테스트 정확도: ', loaded_model.evaluate(X_test, y_test)[1])#평가도 test로. 실제 학습엔 사용하지 않는 데이터이지만 그래도 데이터의 양이 충분하다면 둘은 다른 데이터를 사용하는 것이 바람직하다.


epochs=range(1, len(history.history['acc'])+1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
