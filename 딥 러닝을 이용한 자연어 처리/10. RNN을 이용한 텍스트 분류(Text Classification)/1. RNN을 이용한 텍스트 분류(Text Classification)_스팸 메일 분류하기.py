""" Text Classification은 텍스트가 어떤 종류의 범주에 속하는지 구분하는 것으로, Binary Classification, Multi-Class Classification,
Sentiment Analysis, Intent Analysis(명령, 거절, 질문 등)이 있다. RNN계열의 Vanila RNN, LSTM, GRU등을 학습할 예정이다.

    [1. 케라스를 이용한 텍스트 분류 개요(Text Classification using Keras)]
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))에서 hidden_units은 RNN출력의 크기 즉, 은닉 상태의 크기 자체를 의미한다.
timesteps은 시점의 수로 문서 분류에 사용될 경우 입력 시퀀스에 사용되기에 각 문서에서 단어 수에 해당한다.
input_dim은 입력의 크기로 단어별로 입력될 벡터의 크기 즉, 임베딩 벡터의 차원을 의미한다.
 텍스트 분류는 RNN의 many-to-many에 속하며, 모든 timestep에 입력을 받지만 최종 RNN셀만이 은닉상태가 출력층의 활성화함수를 거쳐 분류하는 과정으로 이루어진다.
 분류 클래스의 개수가 N개라면 출력층에 해당하는 Dense layer의 크기를 N으로 한다. 즉, Dense layer은 그전까지의 입력을 결국 분류해야하는
 클래스의 개수만큼 벡터차원을 가지게 하여 각 클래스별 가능성을 나타내기 위한 전처리?에 해당한다."""
    #[2. 스팸 메일 분류하기(Spam Detection)]
 #1. 스팸 메일 데이터에 대한 이해
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/10.%20RNN%20Text%20Classification/dataset/spam.csv", filename="spam.csv")
data=pd.read_csv('spam.csv', encoding='latin1')
print('총 샘플의 수: ', len(data))
print("(test)상위 5개의 샘플 출력: \n", data[:5])#v1열에 label이, v2열에 content가 있다는 것을 알 수 있다.

#(전처리)현재 csv에는 NaN의 열 3개가 존재하므로 제거하고, ham과 spam을 label로 사용하기 위해 0과 1로 치환한다.
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1']=data['v1'].replace(['ham', 'spam'], [0,1])
print("(test)전처리 후 상위 5개 샘플 출력:\n", data[:5],'\n\n데이터프레임의 정보:')

#데이터프레임의 정보 표시. non-null정보를 통해 Null값의 샘플이 없다는 것을 확인할 수 있다.
data.info()
print('\n결측값 여부: ', data.isnull().values.any(),'\n')

print('(중복확인을 위한) v2열의 유니크한 값: ', data['v2'].unique())#5169(/5572) 총 403개의 중복 샘플 존재.
data.drop_duplicates(subset=['v2'], inplace=True)
print('중복 데이터 제거 후 총 샘플의 수: ', len(data),'\n')

data['v1'].value_counts().plot(kind='bar')
plt.show()#대부분의 메일의 label이 0 즉, 정상 메일임을 의미한다.

print('정상 메일과 스팸 메일의 개수: ')
print(data.groupby('v1').size().reset_index(name='count'))#v1(label)로 그룹나눈 뒤 사이즈를 세는데, 인덱스를 count로 하여 표시한다.
print('정상 메일의 비율: ', round(data['v1'].value_counts()[0]/len(data)*100, 3),'%')#value_counts한 것 중에 값이 0인 count
print('스팸 메일의 비율: ', round(data['v1'].value_counts()[1]/len(data)*100, 3), '%\n')

#데이터의 분리
X_data=data['v2']
y_data=data['v1']
print('(X_data와 y_data의 짝이 맞는지를 확인하기 위한)메일 본문의 개수: ', len(X_data))
print('레이블의 개수: ', len(y_data),'\n')

#***레이블이 굉장히 불균형한 경우(정상87%, 스팸12%) 우연히 테스트데이터에 정상메일만 들어가는 경우가 있을 수 있기에 레이블의 분포가 고르게 되게하는 것이 굉장히 중요하다.
#이때 sklearn의 train_test_split에 인자 stratify로 레이블 데이터를 기재하면 분포를 고려하여 특정 비율로 분리가 가능하다.
X_train, X_test, y_train, y_test=train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)#X와 y데이터를 0.2로 분리하는데, label을 기재하여 따로 비율로 분리한다.
print('--------훈련 데이터의 비율--------')
print('정상 메일=',round(y_train.value_counts()[0]/len(y_train)*100, 3))#well seperated thanks to stratify argument of train_test_split!
print('스팸 메일=',round(y_train.value_counts()[1]/len(y_train)*100, 3))
print('--------테스트 데이터의 비율--------')
print('정상 메일=',round(y_test.value_counts()[0]/len(y_test)*100, 3))
print('스팸 메일=',round(y_test.value_counts()[1]/len(y_test)*100, 3))

#토큰화 과정(전처리)
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded=tokenizer.texts_to_sequences(X_train)
print('(test)integer encoded X_train(상위 5개): ')
print(X_train_encoded[:5], '\n')

#word_to_index확인
word_to_index=tokenizer.word_index
print('word_to_index(Integer encoding 결과 확인): ')
print(word_to_index,'\n')

#***우리가 word_to_index즉, integer encoding을 제작 시 빈도가 낮은 단어는 알아서 삭제해버리는데, 중요한 데이터일 경우 혹은 방대한 양의 데이터의 경우 빈도가 낮은 데이터라고 삭제해버리기 위험하기 때문에
#tokenizer.word_counts.items()를 이용하여 빈도수가 낮은 단어들이 훈련 데이터에서 얼마나 비중을 차지하는지 확인할 수 있기에 이를 통해 삭제할 낮은 빈도수의 기준을 선택하여 삭제할 수 있다.
 #(기준)
threshold=2#이보다 작으면 rare_cnt로 분류한다.
 #(카운트 저장용)
total_cnt=len(word_to_index)#vocab의 전체 단어의 수
rare_cnt=0#frequency가 threshold보다 작은 단어의 개수
 #(빈도 저장용)
total_freq=0#훈련데이터의 전체 단어 빈도수 총 합
rare_freq=0#threshold보다 frequency가 작은 단어의 빈도수 총 합.

for key, value in tokenizer.word_counts.items():#각 단어의 등장빈도(count)_key=단어, value=빈도
    total_freq=total_freq+value

    if(value<threshold):
        rare_cnt=rare_cnt+1
        rare_freq=rare_freq+value
print('등장 빈도가 %s번 이하인 희귀 단어의 수(count): %s' %(threshold-1, rare_cnt))
print('단어 집합에서 희귀 단어의 비율(count): %s' %(rare_cnt/total_cnt*100))#희귀 단어의 비율이 단어 집합에서 무려 55%를 차지한다!
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율(frequency): ', (rare_freq/total_freq)*100, '\n')#하지만 필요한 단어 빈도 측면에서는 6%밖에 차지하지 않는다.
#결론: threshold=2미만인 rare_word에 대하여 무시해도 유효한 데이터를 충분히 얻을 수 있다!
tokenizer=Tokenizer(num_words=total_cnt-rare_cnt+1)#기존에 Tokenizer instantiation시 num_word argument로 반영할 단어의 개수를 지정할 수 있었는데,
#위에서 찾아낸 rare_word의 개수를 total_cnt에서 빼여 num_words인자로 주면 빈도수 기준으로 나뉘기에 원하는 값(threshold=2미만 희귀 단어가 배제된 vocabulary)을 얻을 수 있다.

vocab_size=len(word_to_index)+1
print('rare_word가 제거된 단어집합의 크기: ', vocab_size, '\n')

#적절한 패딩 크기를 정하기 위한 작업
print('메일의 최대 길이: ', max(len(sample) for sample in X_train_encoded))#X_train_encoded의 길이의 max값_ 189!!
print('메일의 평균 길이: ', sum(map(len, X_train_encoded))/len(X_train_encoded))#X_train_encoded에 len을 다 적용하고 sum을 구한 뒤 전체 길이로 나눈 값
plt.hist([len(sample) for sample in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len=max(len(sample) for sample in X_train_encoded)#위에서 출력 시 사용한 값 그대로 다시 구한거임..가독성위해
X_train_padded=pad_sequences(X_train_encoded, maxlen=max_len)#max_len으로 X_train_encoded데이터 padding!
print('\n훈련 데이터의 크기(shape): ', X_train_padded.shape,'\n')


 #[2. RNN으로 스팸 메일 분류하기]
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

embedding_dim=32#Hyper parameters
hidden_units=32

model=Sequential()#기본적인 RNN구조를 항상 생각하자. Embedding layer을 거쳐 RNN으로 들어가고, 최종 timestep에서 Dense layer을 지나 softmax로 들어갔다. 복습하자는 의미에서 8.4의 이미지 RNN학습구조를 다시 첨부하였다.
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])#label; binary classification!
history=model.fit(X_train_padded, y_train, epochs=4, batch_size=64, validation_split=0.2)#validation_split for checking overfitting

#Test Accuracy
X_test_encoded=tokenizer.texts_to_sequences(X_test)#integer encoding
X_test_padded=pad_sequences(X_test_encoded, maxlen=max_len)#padding
print('테스트 정확도: ', model.evaluate(X_test_padded, y_test)[1])#evaluation

epochs=range(1, len(history.history['acc'])+1)
plt.plot(epochs, history.history['loss'])#trainnig loss
plt.plot(epochs, history.history['val_loss'])#test loss
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()#데이터의 양이 적어 overfitting이 빠르게 시작되기에 epochs는 overfitting직전인 3~4가 적당하다는 것을 확인할 수 있다! epochs5가 넘어가면 validation data의 오차가 증가한다.
#우선 내 데이터의 경우 별도로 다운받아 약간은 다른 데이터가 저장된거같은데, 2.0epoch에서 validation의 loss가 낮아지다가 증가하며, epoch3.0에서 낮아진다(overfitting)
#고로 epoch2.0에서 중단하는게 맞아보이고 추후에 새로운 데이터로 trainning시키는 것이 좋을 것 같다.
