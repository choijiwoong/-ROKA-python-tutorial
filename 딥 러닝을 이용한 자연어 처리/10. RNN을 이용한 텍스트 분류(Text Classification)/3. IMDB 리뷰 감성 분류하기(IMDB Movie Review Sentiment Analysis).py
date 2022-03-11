    #[1. IMDB 리뷰 데이터에 대한 이해]
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb#imdb는 앞선 teuters뉴스와 달리 training_data와 test_data를 50:50으로 구준해서 제공해준다.

(X_train, y_train), (X_test, y_test)=imdb.load_data()#(num_words사용 가능)
print('훈련용 리뷰 개수: ', len(X_train))#25000
print('테스트용 리뷰 개수: ', len(X_test))#2500
num_classes=len(set(y_train))
print('카테고리: ', num_classes,'\n')#2(sentiment_bool)

print('(test)첫번째 훈련용 리뷰: ', X_train[0])#integer coded state! (by frequency)
print('(test)첫번째 훈련용 리뷰의 레이블: ', y_train[0])

#리뷰의 길이분포 시각화
reviews_length=[len(review) for review in X_train]
print('리뷰의 최대 길이: ', np.max(reviews_length))
print('리뷰의 평균 길이: ', np.mean(reviews_length))

plt.subplot(1,2,1)
plt.boxplot(reviews_length)#리뷰의 길이를 box로
plt.subplot(1,2,2)
plt.hist(reviews_length, bins=50)
plt.show()

#실제 레이블의 분포 확인
unique_elements, counts_element=np.unique(y_train, return_counts=True)
print('각 레이블에 대한 빈도수: ')
print(np.asarray((unique_elements, counts_element)), '\n')#레이블 별 12500, 12500데이터를 갖고 있다는 것을 확인(균등)

#word_to_index, index_to_word
word_to_index=imdb.get_word_index()#규칙 상 index+3을 해야 실제 매핑되는 정수가 나온다.
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value+3]=key#실제 가용한 index_to_word (without rule of imdb)
print('(test)빈도수 상위 1등 단어: ', index_to_word[4])
print('(test)빈도수 상위 3938등 단어: ', index_to_word[3941], '\n')

#첫번째 훈련용 데이터리뷰의 각 단어들 decoding.
for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index]=token
print('첫번째 훈련용 데이터 리뷰의 각 단어들: ')
print(' '.join([index_to_word[index] for index in X_train[0]]))


    #[2. GRU로 IMDB 리뷰 감성 분류하기]
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

vocab_size=10000#단어집합 크기 제한(빈도수)
max_len=500#for padding

(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=vocab_size)
X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)

#many-to-one! GRU_GRU도 마찬가지로 LSTM의 구조를 단순화 시켰을 뿐 성능은 비슷함. 그래도 최대한 많은 예제를 보여주려는 이 책이 참 좋네
embedding_dim=100
hidden_units=128

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(GRU(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history=model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.2)

loaded_model=load_model('GRU_model.h5')
print('\n테스트 정확도: ', loaded_model.evaluate(X_test, y_test)[1])#88.93%!


#사용(리뷰의 sentiment classification)을 위해서는 임의의 문장에 대해 전처리가 필요하다.
def sentiment_predict(new_sentence):
    new_sentence=re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()#알파벳, 숫자, 공백 제외 다 지우고 lower
    encoded=[]

    for word in new_sentence.split():
        try:
            if word_to_index[word]<=10000:
                encoded.append(word_to_index[word]+3)
            else:
                encoded.append(2)#10000이상의 숫자는 <unk>토큰에 append
        except KeyError:#이상한 단어가 들어왔을시 <unk>토큰에 append
            encoded.append(2)
    pad_sequence=pad_sequences([encoded], maxlen=max_len)#integer encoding된 단어를 padding
    score=float(loaded_model.predict(pad_sequence))#전처리된 new_sentence를 모델에 넣어 얻은 예측값을 저장.

    if(score>0.5):
        print(score, '%로 긍정인 리뷰입니다.')
    else:
        print(score, '%로 부정인 리뷰입니다.')
test_bad_review="This movie was just way too overrated. The fighting was not professional and in slow motion. I was expecting more from a 200 million budget movie. The little sister of T.Challa was just trying too hard to be funny. The story was really dumb as well. Don't watch this movie if you are going because others say its great unless you are a Black Panther fan or Marvels fan."
sentiment_predict(test_bad_review)#0.76

test_positive_review=" I was lucky enough to be included in the group to see the advanced screening in Melbourne on the 15th of April, 2012. And, firstly, I need to say a big thank-you to Disney and Marvel Studios. \
Now, the film... how can I even begin to explain how I feel about this film? It is, as the title of this review says a 'comic book triumph'. I went into the film with very, very high expectations and I was not disappointed. \
Seeing Joss Whedon's direction and envisioning of the film come to life on the big screen is perfect. The script is amazingly detailed and laced with sharp wit a humor. The special effects are literally mind-blowing and the action scenes are both hard-hitting and beautifully choreographed."
sentiment_predict(test_positive_review)#0.99
