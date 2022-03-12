    #[1. 네이버 영화 리뷰 데이터에 대한 이해와 전처리]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

 #1. 데이터 로드하기
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data=pd.read_table('ratings_train.txt')
test_data=pd.read_table('ratings_test.txt')

print('(데이터 로드하기)\n훈련용 리뷰 개수: ', len(train_data))
print('(test)상위 5개 훈련용 리뷰 출력:', train_data[:5])#id는 sentiment classification에 도움이 안되는 데이터임을 알 수 있다!
print('(test)상위 5개 테스트용 리뷰 출력:', test_data[:5],'\n\n')#또한 한국어 특성상 띄어쓰기도 잘 안돼있고 오타도 많고 신조어, 줄임말도 많음을 알 수 있다. 좆됬다 근데 천재가 이미 만들어놓은거 전에 배웠다ㅋ

 #2. 데이터 정제하기
print('\n\n(데이터 정제하기)\n훈련데이터의 중복유무 확인: ', train_data['document'].nunique(), ', 훈련데이터 label 중복유무 확인??ㅋㅋ왜하노: ', train_data['label'].nunique())
#무튼 총 150000샘플이 train_data로 존재하는데, 중복제거시 146182개니까 총 4000개정도의 중복샘플이 존재한다는 것이기에 삭제한다.
train_data.drop_duplicates(subset=['document'], inplace=True)#document열 기준 중복되는 데이터 제거
print('\ndocument열 기준 중복 제거 후 총 샘플의 수:', len(train_data),'\n')

#데이터 분포 확인
train_data['label'].value_counts().plot(kind='bar')
plt.show()#균일해보인다.
print(train_data.groupby('label').size().reset_index(name='count'),'\n')#label0이 근소하게 많다

print('리뷰중에 Null값 유무:', train_data.isnull().values.any(),'\n')
print('Null이 어떤 열에 존재하는지 확인: \n', train_data.isnull().sum(),'\n')#만약 필요없는 id열이 null이면 굳이 처리할 필요가 없기에. 하지만 document에 1개 확인
print("document열에 Null값을 가진 샘플이 어느 index에 위치해있는지: \n", train_data.loc[train_data.document.isnull()],'\n')#document null인 항목(document)의 location.

train_data=train_data.dropna(how='any')#위에 어느 인덱스고 어느 열이고는 그냥 공부 차원에서 해본거고 null있는거 다 삭제
print('dropna실행 후 null값이 존재하는지 확인: ', train_data.isnull().values.any())
print('null값 제거 후 총 샘플의 개수: ', len(train_data),'\n')

#특수문자를 없애는데에 영어 re.sub(r'[^a-zA-Z ]', '', eng_text)와 유사하게 처리가 가능하다.
train_data['document']=train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
print("(test)regex이용 한글만 남긴 상위 5개 리뷰:\n", train_data[:5],'\n')

#만약 영어로만 이루어진 리뷰일 경우 한국어만 필터링하면서 NaN값이 되었을 가능성이 농후하니 null값을 다시 확인한다.
train_data['document']=train_data['document'].str.replace('^ +', "")#double space이상을 empty value로 변경
train_data['document'].replace('', np.nan, inplace=True)#empty value를 NaN으로 변경(두번에 걸친 이유는 일반 space마저도 empty value로 바꾸면 정상문장 사이 띄어쓰기가 사라지기에)
print('한글 필터링 이후 null값의 유무 확인:')
print(train_data.isnull().sum(),'\n')#789개 null데이터 형성.

print("(test)Null이 있는 상위 5개 행 출력: ")
print(train_data.loc[train_data.document.isnull()][:5])
#제거
train_data=train_data.dropna(how='any')
print("한글 필터링 후 null값 제거한 뒤의 총 샘플의 개수: ", len(train_data),'\n\n')#145393

#(놀랍게도 이 테스트데이터 전처리를 빼먹고 지나가서 9분 걸린 아래 코드를 다시 컴파일 해야댐...ㅠ)
test_data.drop_duplicates(subset=['document'], inplace=True)#document중복제거
test_data['document']=test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
test_data['document']=test_data['document'].str.replace('^ +', "")
test_data['document'].replace('', np.nan, inplace=True)
test_data=test_data.dropna(how='any')
print('전처리 후 테스트용 샘플의 개수: ', len(test_data))

 #3. 토큰화_java.lang.java.lang.ExceptionInInitializerError: java.lang.ExceptionInInitializerError으로 지금부터 Colab사용
stopwords=['의 ', '가', '이', '은', '들', '는', '좀', '잘',' 걍', '과', '도', '를', '으로', '자', '에 ','와 ','한 ','하다']#대충 이정도만 사용
okt=Okt()
print("\n\n(토큰화)\n(test)okt 형태소분석기 테스트:", okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem=True))

#토근화, 불용어제거
X_train=[]
for sentence in tqdm(train_data['document']):
    tokenized_sentence=okt.morphs(sentence, stem=True)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
    X_train.append(stopwords_removed_sentence)
print('\n(test)토큰화, 불용어제거된 상위3개의 샘플: ')
print(X_train[:3])

X_test=[]#테스트데이터도 같이적용
for sentence in tqdm(test_data['document']):
    tokenized_sentence=okt.morphs(sentence, stem=True)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

 #4. 정수 인코딩
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)
print("\n\n(정수 인코딩)\nvocabulary: ", tokenizer.word_index)#매우 낮은 빈도수도 포함되어있음.
#빈도수 적은 희귀단어 삭제 전, 얼만큼의 비중을 차지하는지 신중하게 확인
threshold=3#standard
total_cnt=len(tokenizer.word_index)#vocab_size
rare_cnt=0#rare_word under threshold
total_freq=0#sum(words.frequency)
rare_freq=0#sum(rare_words.frequency)이해하지?

for key, value in tokenizer.word_counts.items():#단어들을 각각 count하여 items로 key,value에 저장(단어, 빈도수)
    total_freq=total_freq+value

    if (value<threshold):
       rare_cnt=rare_cnt+1
       rare_freq=rare_freq+value
print('단어 집합(vocabulary)의 크기:', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold-1, rare_cnt))#threshold미만==threshold-1이하!
print('단어 집합에서 희귀 단어의 비율:',(rare_cnt/total_cnt)*100)
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율:', (rare_freq/total_freq)*100)
#결론: 전체집합에서의 비율을 절반을 넘지만, 빈도비율은 2%가 안된다! 배제해도 된다.

vocab_size=total_cnt-rare_cnt+1
print('rare_word를 제거한 언어 집합의 크기: ', vocab_size)
tokenizer=Tokenizer(vocab_size)#tokenizer 재설정_(num_word=vocab_size)
tokenizer.fit_on_texts(X_train)#재설정된 tokenizer로 data integer encoding
X_train=tokenizer.texts_to_seqeunces(X_train)
X_test=tokenizer.texts_to_sequences(X_test)

print("(test)num_word설정된 tokenizer이용한 X_train상위 3개 샘플:")
print(X_train[:3])

y_train=np.array(train_data['label'])#label값은 별도의 전처리가 필요없음
y_test=np.array(test_data['label'])

 #5. 빈 샘플(empty samples)제거_빈도수가 낮은 단어가 삭제되었다!(by num_word argument of tokenizer)는 빈도수가 낮은 단어만으로 구성된 샘플의 empty가능성(와..존나게 세심하고 조심하고 또 확인하고 해야하네..ㄷㄷ)
drop_train=[index for index, sentence in enumerate(X_train) if len(sentence)<1]#길이가 0인 X_train의 sentence들의 index저장
X_train=np.delete(X_train, drop_train, axis=0)
y_train=np.delete(y_train, drop_train, axis=0)#그에 해당되는 y_label까지 삭제하기 위해 drop_train에 index를 저장함.
print('\n\n(빈 샘플제거)\nnum_words적용된 tokenizer처리 후 빈 샘플 제거된 X_train샘플의 개수:', len(X_train), ", y_train개수", len(y_train))

 #6. 패딩
print('\n\n(패딩)\n리뷰의 최대 길이: ', max(len(review) for review in X_train))#69(내껀70)
print('리뷰의 평균 길이: ', sum(map(len, X_train))/len(X_train))#10(내껀11. 뭔가 전처리 하나 안된거같은데 일단 패숭!)
plt.hist([len(review) for review in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()#분포로 보아 패딩을 위한 최적의 max_len값을 특정하기 어렵다. 고로 max_len이하의 샘플의 비율이 몇%인지를 확인해보자.

def below_threshold_len(max_len, nested_list):
    count=0
    for sentence in nested_list:
        if(len(sentence)<=max_len):
            count=count+1
    print('전체 샘플 중 길이가 %s이하인 샘플의 비율: %s'%(max_len, (count/len(nested_list))*100))
max_len=40
print("max_len=40 이하인 샘프의 비율: ", below_threshold_len(max_len, X_train))#97.6%
max_len=30
print("max_len=30 이하인 샘플의 비율: ", below_threshold_len(max_len, X_train))#93.6%
max_len=20
print("max_len=20 이하인 샘플의 비율: ", below_threshold_len(max_len, X_train))#87.3%
#max_len=30으로 결정! 94.3%비율로 적당한 로스라고 판단.
X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)


    #[2. LSTM으로 네이버 영화 리뷰 감성 분류하기]
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=100
hidden_units=128

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)#loss기준 check
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])#Sentiment Classification!
history=model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model=load_model('best_model.h5')
print('\n테스트 정확도: ', loaded_model.evaluate(X_test, y_test)[1])#왜 1로 접근하느지는 확인해봐야겠다.


    #[3. 리뷰 예측해보기]
def sentiment_predict(new_sentence):#전처리+predict
    new_sentence=re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', new_sentence)
    new_sentence=okt.morphs(new_sentence, stem=True)
    new_sentence=[word for word in new_sentence if not word in stopwords]
    
    encoded=tokenizer.texts_to_sequences([new_sentence])
    pad_new=pad_sequences(encoded, maxlen=max_len)
    
    score=float(loaded_model.predict(pad_new))
    if(score>0.5):
        print(score,'확률로 긍정리뷰입니다.')
    else:
        print(100-score,'확률로 부정리뷰입니다.')#predict_score의 확률의 의미보단 내가 봤을때 정확도 따지는게 좋을거같아서 조금 바꿈!
print("이 영화 존나 재밌네ㅋㅋㅋㅋㅋ이게 맞나: ", sentiment_predict("이 영화 존나 재밌네ㅋㅋㅋㅋㅋ이게 맞나"))
print("아 씨발 좆같네 이걸 영화라고 만드냐 병신들아ㅗㅗ: ", sentiment_predict("아 씨발 좆같네 이걸 영화라고 만드냐 병신들아ㅗㅗ"))
print("좀 애매한데 좋긴한데 뭔가 엔딩이 맘에 안들어..근데 또 자꾸 생각나긴하네: ", sentiment_predict("좀 애매한데 좋긴한데 뭔가 엔딩이 맘에 안들어..근데 또 자꾸 생각나긴하네"))
