""" BiLSTM은 두 개의 독립적인 LSTM아키텍처를 사용하는 구조로 뒤의 문맥까지 고려하기 위하여 역방향 LSTM셀을 사용한다.
일반적인 many-to-many의 BiLSTM은 단순한데, many-to-one에 사용하면 역방향 LSTM도 순방향과 같은 시점의 은닉상태를 사용하느냐 하면 그렇지 않고
말 그대로 방향이 다르니게 순방향LSTM은 마지막시점의 은닉 상태를, 역방향 LSTM은 첫번째 시점의 은닉상태를 반환한다."""
    #[1. 스팀 리뷰 데이터에 대한 이해와 전처리]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

 #1. 데이터 로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename="steam.txt")
total_data=pd.read_table('steam.txt', names=['label', 'reviews'])
print("전체 리뷰 개수: ", len(total_data))
print('(test)상위 5개 샘플:', total_data[:5])

print("각 열에 대하여 중복을 제외한 샘플의 수:", total_data['reviews'].nunique(), total_data['label'].nunique())#중복확인
#중복제거
total_data.drop_duplicates(subset=['reviews'], inplace=True)
print('중복제거 후 총 샘플의 수:', len(total_data))

print("Null값 유무:", total_data.isnull().values.any())#False

 #2. 훈련 데이터와 테스트 데이터 분리하기
train_data, test_data=train_test_split(total_data, test_size=0.25, random_state=42)
print('\n훈련용 리뷰의 개수: ', len(train_data))
print('테스트용 리뷰의 개수: ', len(test_data))

 #3. 레이블의 분포 확인
train_data['label'].value_counts().plot(kind='bar')
print('\n레이블의 분포의 정확한 수치(value_counts()): ')
print(train_data.groupby('label').size().reset_index(name='count'))

 #4. 데이터 정제하기
train_data['reviews']=train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['reviews'].replace('', np.nan, inplace=True)
print("regex이용 한글 필터링 후 null값 확인: ", train_data.isnull().sum())#0!
#테스트테이터에도 같은 과정
test_data.drop_duplicates(subset=['reviews'], inplace=True)
test_data['reviews']=test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
test_data['reviews'].replace('', np.nan, inplace=True)
test_data=test_data.dropna(how='any')
#불용어정의_(당연히 게임관련이라 게임이 빠진게 신기)
stopwords=['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']

 #5. 토큰화
mecab=Mecab()

train_data['tokenized']=train_data['reviews'].apply(mecab.morphs)
train_data['tokenized']=train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized']=test_data['reviews'].apply(mecab.morphs)
test_data['tokenized']=test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

 #6. 단어와 길이 분포 확인하기
#단어분포확인
negative_words=np.hstack(train_data[train_data.label==0]['tokenized'].values)#형식 잘 확인!
positive_words=np.hstack(train_data[train_data.label==1]['tokenized'].values)

negative_word_count=Counter(negative_words)
print("\n가장 흔한 부정단어(by Counter):", negative_word_count.most_common(20))
positive_word_count=Counter(positive_words)
print("가장 흔한 긍정단어(by Counter):", positive_word_count.most_common(20),'\n')

#길이분포확인
fig, (ax1, ax2)=plt.subplots(1,2,figsize=(10,5))

text_len=train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))#map(function!)
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이: ', np.mean(text_len))#14.9

text_len=train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이: ', np.mean(text_len))#15.2
plt.show()#유의미한 차이는 없다.

X_train=train_data['tokenized'].values
y_train=train_data['label'].values
X_test=test_data['tokenized'].values
y_test=test_data['label'].values

 #7. 정수 인코딩
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)

#num_words전 확인
threshold=2
total_cnt=len(tokenizer.word_index)
rare_cnt=0
total_freq=0
rare_freq=0

for key, value in tokenizer.word_counts.items():#단어, 빈도수
    total_freq=total_freq+value
    if(value<threshold):
        rare_cnt=rare_cnt+1
        rare_freq=rare_freq+value
print('단어 집합(vocabulary)의 크기: ', total_cnt)#32817
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold-1, rare_cnt))
print('단어 집합에서 희귀 단어의 비율: ', (rare_cnt/total_cnt)*100)#42%
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율: ', (rare_freq/total_freq)*100)#1.2%

vocab_size=total_cnt-rare_cnt+2#(OOV토큰 고려)
print('rare_word제거 후 단어 집합의 크기: ', vocab_size)#18941

tokenizer=Tokenizer(vocab_size, oov_token='OOV')#정수 인코딩 과정에서 큰 숫자가 부여된 단어들은 OOV로 변환
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)

print('(test) Integer encoded X_train상위 3개:')
print(X_train[:3])
print('(test) Integer encoded X_test상위 3개:')
print(X_test[:3])

 #8. 패딩
print('리뷰의 최대 길이: ', max(len(review) for review in X_train))
print('리뷰의 평균 길이: ', sum(map(len, X_train))/len(X_train))
plt.hist([len(review) for review in X_train], bins=50)#길이를 hist에 등록
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
    count=0
    for sentence in nested_list:
        if(len(sentence)<=max_len):
            count=count+1
    print('전체 샘플 중 길이가 %s이하인 샘플의 비율: %s'%(max_len, (count/len(nested_list))*100))
max_len=60
below_threshold_len(max_len, X_train)#99.99%
max_len=50
below_threshold_len(max_len, X_train)#99.79%

X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)

    #[2. BiLSTM으로 스팀 리뷰 감성 분류하기]
import re
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=100
hidden_units=128

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(hidden_units)))#Bidirectional LSTM. Bidirectional을 recursive구조?처럼 놓아 사용한다는거 기억하면 될듯
model.add(Dense(1, activation='sigmoid'))

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint("best_model.h5", monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history=model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=256, validation_split=0.2)

loaded_model=load_model('best_model.h5')
print('\n테스트 정확도: %.4f'%(loaded_model.evaluate(X_test, y_test)[1]))

 #리뷰 예측해보기
def sentiment_predict(new_sentence):
    new_sentence=re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence=mecab.morphs(new_sentence)
    new_sentence=[word for word in new_sentence if not word in stopwords]

    encoded=tokenizer.texts_to_sequences([new_sentence])
    pad_new=pad_sequences(encoded, maxlen=max_len)

    score=float(loaded_model.predict(pad_new))
    if (score>0.5):
        print('{:.2f}% 확률로 긍정리뷰입니다.'.format(score*100))
    else:
        print("{:.2f}% 확률로 부정리뷰입니다.".format((1-score)*100))
sentiment_predict("이 게임 개발자 현재 수온체크중이었으면")
sentiment_predict("ㄹㅇ 인생게임ㅋㅋㅋ시간 살살녹노!")
sentiment_predict("좆같이 재밌네 진짜 이거만 하다가 인생 망할거같이 재밌다 오늘하루도 한게없네ㅠㅠ")
