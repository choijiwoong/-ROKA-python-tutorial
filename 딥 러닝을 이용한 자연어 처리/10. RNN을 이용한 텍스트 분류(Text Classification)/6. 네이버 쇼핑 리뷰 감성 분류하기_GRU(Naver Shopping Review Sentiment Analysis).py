    #[1. 네이버 쇼필 리뷰 데이터에 대한 이해와 전처리]
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

 #1. 데이터 로드하기
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
total_data=pd.read_table('ratings_total.txt', names=['ratings', 'review'])
print('(데이터 로드하기)\n전체 리뷰 개수: ', len(total_data))#200,000
print('(test)상위 5개 샘플 출력:\n',total_data[:5])#ratings에 평점1,2,4,5로 작성되어있음. 감성분류를 위한 전처리 필요.

 #2. 훈련 데이터와 테스트 데이터 분리하기
#평점의 label화
total_data['label']=np.select([total_data.ratings>3], [1], default=0)#total_data의 ratings이 3초과일 경우에 1값으로 저장하고 default는 =으로
print('\n\n(훈련 데이터와 테스트 데이터 분리하기)\n(test)ratings의 label화 이루 상위 5개 샘플 출력:\n', total_data[:5])
#중복제거
print("중복제외 샘플개수 카운트 ", total_data['ratings'].nunique(), total_data['review'].nunique(), total_data['label'].nunique())#review에서 중복데이터확인(약 100개)
total_data.drop_duplicates(subset=['review'], inplace=True)
print('reviews열의 중복제거 후 총 샘플의 수:', len(total_data))
#null값처리
print('total_data의 null값 유무 확인: ', total_data.isnull().values.any())#False!!!!!null값 처리 불필요ㅎ
#데이터 분리
train_data, test_data=train_test_split(total_data, test_size=0.25, random_state=42)
print('훈련용 리뷰의 개수: ', len(train_data))
print('테스트용 리뷰의 개수: ', len(test_data))

 #3. 레이블의 분포 확인
train_data['label'].value_counts().plot(kind='bar')
plt.show()#비슷해보임
print("\n\n(레이블의 분포 확인)\n", train_data.groupby('label').size().reset_index(name='count'))#약간의 차이(대략 label=1이 100개더 많음) 50:50으로 볼 수 있다. (우리 이전에 스팸에서 차이많이나면 split에 label인자로 줬었음! 물론 지금은 필요X)

 #4. 데이터 정제하기
train_data['review']=train_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
train_data['review'].replace('', np.nan, inplace=True)
print("\n\n(데이터 정제하기)\nregex이용 한글만 남겨준 후 null값 확인", train_data.isnull().sum())#False!
#테스트데이터에도 이전의 과정 적용
test_data['review']=test_data['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", '')
test_data=test_data.dropna(how='any')
print('전처리 후 테스트용 샘플의 개수:' ,len(test_data))

 #5. 토큰화 & 불용어제거
mecab=Mecab()
print('\n\n(토큰화 & 불용어제거)\n(test)임의의 문장 토큰화 테스트: ', mecab.morphs("와 이런 것도 상품이라고 차라리 내가 만들게"))

stopwords=['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
train_data['tokenized']=train_data['review'].apply(mecab.morphs)
train_data['tokenized']=train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
test_data['tokenized']=test_data['review'].apply(mecab.morphs)
test_data['tokenized']=test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

 #6. 단어와 길이 분포 확인하기
#긍정리뷰와 부정리뷰에 어떤 단어들이 많이 등장하는지
negative_words=np.hstack(train_data[train_data.label==0]['tokenized'].values)#hstack은 두 배열을 가로로 이어붙인다.
positive_words=np.hstack(train_data[train_data.label==1]['tokenized'].values)

negative_word_count=Counter(negative_words)#Counter사용
print('\n\n(단어와 길이 분포 확인하기)\n부정리뷰단어중 가장 빈도수 높은 상위20개 단어 출력:', negative_word_count.most_common(20))
positive_word_count=Counter(positive_words)#Counter사용
print('긍정리뷰단어중 가장 빈도수 높은 상위20개 단어 출력:', positive_word_count.most_common(20),'\n')

#긍정리뷰와 부정리뷰의 길이는 어떻게 분포되는지(패딩을 위해선 아님 패딩보려면 max_len도 봤을거임)
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

text_len=train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이: ', np.mean(text_len))

text_len=train_data[train_data['label']==0]['tokenized'].map(lambda x:len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
ax2.set_title('Negative Reviews')
fig.suptitle('words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이: ', np.mean(text_len))
plt.show()#긍정리뷰보다 부정리뷰가 좀 더 긴 경향이 있다는 것을 확인할 수 있다.(조금)

X_train=train_data['tokenized'].values
y_train=train_data['label'].values
X_test=test_data['tokenized'].values
y_test=test_data['label'].values

 #7. 정수 인코딩
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)#이전 챕터와 같은 오류가 발생해서 일단 진행하기로함. AttributeError: 'int' object has no attribute 'lower'
#라고 떠서 전처리도 확인하고 값도 확인하고 했는데도 그러네.
#아 씨ㅣ...후....위에 del duplicate두번 써뒀네...작자가...후..무튼 해결
#아 뭐야 왜 또 똑같은 오ㅠ따ㅓㅏ더ㅓㅔ어메ㅓㅇ
#ㅋㅋㅋㅋ문제점을 찾아버렸다 그냥 colab이어서였다. 처음부터 다시 싹 다 컴파일하니 되네 씨발거

#등장횟수 1인 단어 배제 전에 비중 확인
threshold=2
total_cnt=len(tokenizer.word_index)
rare_cnt=0
total_freq=0
rare_freq=0

for key, value in tokenizer.word_counts.items():
    total_freq=total_freq+value
    if (value<threshold):
        rare_cnt=rare_cnt+1
        rare_freq=rare_freq+value
print('\n\n(정수 인코딩)\n단어 집합(vocabulary)의 크기: ', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold-1, rare_cnt))
print('단어 집합에서 희귀 단어의 비율: ', (rare_cnt/total_cnt)*100)#45.5%
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율: ', (rare_freq/total_freq)*100)#0.8% 중요하지 않음.

vocab_size=total_cnt-rare_cnt+2#OOV토큰이 있으므로 +1(데이터 특성상 그런게 있나봄. 저번에 <sen>같은 태그처럼)
print('\n희귀단어(빈도수1이하)인 단어 제거 후 단어집합의 크기: ', vocab_size)

tokenizer=Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)

print('(test)희귀단어 제거된 X_train의 상위 3개 샘플:',X_train[:3])
print('(test)희귀단어 제거된 X_test의 상위 3개 샘플: ', X_test[:3])

 #8. 패딩
print('리뷰의 최대 길이:', max(len(review) for review in X_train))#85
print('리뷰의 평균 길이:', sum(map(len, X_train))/len(X_train))#15
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

#적절한 패딩 값 찾기
def below_threshold_len(max_len, nested_list):
    count=0
    for sentence in nested_list:
        if(len(sentence)<=max_len):
            count=count+1
    print('전체 샘플 중 길이가 %s이하인 샘플의 비율: %s'%(max_len, (count/len(nested_list))*100))

max_len=80
below_threshold_len(max_len, X_train)#99.9993
max_len=70
below_threshold_len(max_len, X_train)#99.9979이라서 솔직히 더 자르고 싶은데 원래 데이터는 보존해야하는거니까...정말 메모리 없거나 시간없거나 할때 padding_size조절하자.
X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)


    #[3. GRU로 네이버 쇼핑 리뷰 감성 분류하기]
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=100
hidden_units=128

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history=model.fit(X_train, y_train, epochs=15, callbacks=[es,mc], batch_size=64, validation_split=0.2)

loaded_model=load_model('best_model.h5')
print('\n테스트 정확도: ', loaded_model.evaluate(X_test, y_test)[1])

    #[4. 리뷰 예측해보기]
def sentiment_predict(new_sentence):
    new_sentence=re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','',new_sentence)
    new_sentence=mecab.morphs(new_sentence)
    new_sentence=[word for word in new_sentence if not word in stopwords]

    encoded=tokenizer.texts_to_sequences([new_sentence])
    pad_new=pad_sequences(encoded, maxlen=max_len)

    score=float(loaded_model.predict(pad_new))
    if(score>0.5):
        print(score, '확률로 긍정리뷰입니다.')
    else:
        print(1-score, '확률로 부정리뷰입니다.')
sentiment_predict('이거 정말 좆같아요!')
sentiment_predict('이거 정말 좋아요!')
sentiment_predict('이거 정말 존나게 비싸고 냄새나는데 좀 맘에들기도 하네요. 이런 좆같은 쓰레기 냄새가 내 취향이었을줄은 몰랐어요 감사해요')
