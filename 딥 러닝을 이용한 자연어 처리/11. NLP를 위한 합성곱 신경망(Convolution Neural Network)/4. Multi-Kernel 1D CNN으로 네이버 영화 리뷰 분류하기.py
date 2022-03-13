 #1. 네이버 영화 데이터 수집 & 전처리는 이전의 챕터와 동일하게 수행한다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data=pd.read_table('ratings_train.txt')#read_table
test_data=pd.read_table('ratings_test.txt')
#train_data
train_data.drop_duplicates(subset=['document'], inplace=True)#del duplication
train_data=train_data.dropna(how='any')#drop null
train_data['document']=train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")#regex filtering
train_data['document']=train_data['document'].str.replace("^ +", "")#del double space
train_data['document'].replace('', np.nan, inplace=True)#remove NaN
train_data=train_data.dropna(how='any')
#test_data
test_data.drop_duplicates(subset=['document'], inplace=True)
test_data=test_data.dropna(how='any')
test_data['document']=test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", '')
test_data['document']=test_data['document'].str.replace("^ +", '')
test_data['document'].replace('', np.nan, inplace=True)
test_data=test_data.dropna(how='any')

#Tokenize
stopwords=['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt=Okt()
X_train=[]
for sentence in train_data['document']:
    tokenized_sentence=okt.morphs(sentence, stem=True)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
    X_train.append(stopwords_removed_sentence)
X_test=[]
for sentence in test_data['document']:
    tokenized_sentence=okt.morphs(sentence, stem=True)
    stopwords_removed_sentence=[word for word in tokenized_sentence if not word in stopwords]
    X_test.append(stopwords_removed_sentence)

#Integer encoding
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold=3
total_cnt=len(tokenizer.word_index)
rare_cnt=0
total_freq=0
rare_freq=0
for key, value in tokenizer.word_counts.items():
    total_freq=total_freq+value
    if(value<threshold):
        rare_cnt=rare_cnt+1
        rare_freq=rare_freq+value
vocab_size=total_cnt-rare_cnt+1

tokenier=Tokenizer(vocab_size)
tokenizer.fit_on_texts(X_train)
X_train=tokenizer.texts_to_sequences(X_train)
X_test=tokenizer.texts_to_sequences(X_test)

y_train=np.array(train_data['label'])
y_test=np.array(test_data['label'])

#Drop empty one more(because of num_words)
drop_train=[index for index, sentence in enumerate(X_train) if len(sentence)<1]
X_train=np.delete(X_train, drop_train, axis=0)
y_train=np.delete(y_train, drop_train, axis=0)
drop_test=[index for index, sentence in enumerate(X_test) if len(sentence)<1]#test데이터도 마찬가지 수행
X_test=np.delete(X_test, drop_test, axis=0)
y_test=np.delete(y_train, drop_test, axis=0)
#Padding
X_train=pad_sequences(X_train, 30)
X_test=pad_sequences(X_test, 30)


 #2. Multi-Kernel 1D CNN으로 네이버 영화 리뷰 분류하기_다수의 커널을 사용할 경우 Functional API로 구현한다.
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

embedding_dim=128
dropout_ratio=(0.5, 0.8)
num_filters=128
hidden_units=128

model_input=Input(shape=(max_len,))#as padding size
z=Embedding(vocab_size, embedding_dim, input_length=max_len, name='embedding')(model_input)#as Functional API
z=Dropout(dropout_ratio[0])(z)

conv_blocks=[]
for sz in [3,4,5]:#functional API의 장점. for문사용가능
    conv=Conv1D(filters=num_filters, kernel_size=sz, padding='valid', activation='relu', strides=1)(z)
    conv=GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)#kernel_size별 Pooled scalar를 append

z=Concatenate()(conv_blocks) if len(conv_blocks)>1 else conv_blocks[0]
z=Dropout(dropout_ratio[1])(z)
z=Dense(hidden_units, activation='relu')(z)
model_output=Dense(1, activation='sigmoid')(z)

model=Model(model_input, model_output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('CNN_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=2, callbacks=[es, mc])

loaded_model=load_model("CNN_model.h5")
print('\n 테스트 정확도: %4.f'%(loaded_model.evaluate(X_test, y_test)[1]))


 #3. 리뷰 예측해보기_동일한 sentiment function
def sentiment_predict(new_sentence):
  new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
  new_sentence = okt.morphs(new_sentence, stem=True)
  new_sentence = [word for word in new_sentence if not word in stopwords]
  encoded = tokenizer.texts_to_sequences([new_sentence])
  pad_new = pad_sequences(encoded, maxlen = max_len)
  score = float(loaded_model.predict(pad_new))
  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
sentiment_predict('이 영화 핵노잼 ㅠㅠ')
sentiment_predict('이딴게 영화냐 ㅉㅉ')
sentiment_predict('감독 뭐하는 놈이냐?')
sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')
