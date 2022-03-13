 #1. 데이터에 대한 전처리는 11-2 RNN을 이용한 텍스트 분류 챕터의 스팸 메일 분류하기와 동일하게 수행하였다고 가정.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/10.%20RNN%20Text%20Classification/dataset/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin1')

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
data.drop_duplicates(subset=['v2'], inplace=True)
X_data = data['v2']
y_data = data['v1']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train)
word_to_index = tokenizer.word_index

threshold = 2
total_cnt = len(word_to_index)
rare_cnt = 0
total_freq = 0 
rare_freq = 0
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
tokenizer = Tokenizer(num_words = total_cnt - rare_cnt + 1)


vocab_size = len(word_to_index) + 1
X_train_padded = pad_sequences(X_train_encoded, maxlen = 189)

 #2. 1D CNN으로 스팸 메일 분류하기
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=32
dropout_ratio=0.3
num_filters=32
kernel_size=5

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history=model.fit(X_train_padded, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es, mc])

X_test_encoded=tokenizer.texts_to_sequences(X_test)
X_test_padded=pad_sequences(X_test_encoded, maxlen=max_len)
print('\n테스트 정확도: ', model.evaluate(X_test_padded, y_test)[1])
