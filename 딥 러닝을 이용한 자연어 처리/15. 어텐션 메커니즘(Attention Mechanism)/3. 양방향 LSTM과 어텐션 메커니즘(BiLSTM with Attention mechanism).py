 #1. IMDB 리뷰 데이터 전처리하기
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=10000#제한
(X_train, y_train), (X_test, y_test)=imdb.load_data(num_words=vocab_size)

#for padding
print('리뷰의 최대 길이: ', max(len(l) for l in X_train))#2494
print('리뷰의 평균 길이: ', sum(map(len, X_train))/len(X_train))#238
max_len=500
X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)

 #2. 바다나우 어텐션(Bahdanau Attention)_텍스트 분류에서 어텐션을 사용하는 이유는 RNN의 Vanising Gradient에서 발생한 손실을 다시 참고하기 위함이다.
import tensorflow as tf

class BahdanauAttention(tf.keras.Model):#큰일났다!!! Sequential API, Function API, 다음 Class API다!!!
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()#(lstm, hidden_state)
        self.W1=Dense(units)#Encoder각 timestep에 곱해질 학습가능한 행렬1
        self.W2=Dense(units)#Decoder t-1 hidden_state에 곱해질 학습가능한 행렬2
        self.V=Dense(1)#위 두개를 곱한값을 더하여 tanh연산뒤에 곱해질 학습가능한 행렬3

    def call(self, values, query):#query.shape==(batch_size, hidden_size)
        hidden_with_time_axis=tf.expand_dims(query, 1)#가중치 연산 후 덧셈을 위한 차원조정.
        score=self.V(tf.nn.tanh(self.W1(values)+self.W2(hidden_with_time_axis)))#행렬3(tanh(행렬1,행렬2))
        attention_weights=tf.nn.softmax(score, axis=1)#Attention Distribution by softmax

        context_vector=attention_weights*values
        context_vector=tf.reduce_sum(context_vector, axis=1)#각 timestep(encoder)의 scalar값?

        return context_vector, attention_weights

 #3. 양방향 LSTM+어텐션 메커니즘(BiLSTM with Attention Mechanism)
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os

sequence_input=Input(shape=(max_len,), dtype='int32')
embedded_sequences=Embedding(vocab_size, 128, input_length=max_len, mask_zero=True)(sequence_input)
lstm=Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_sequences)
lstm, forward_h, forward_c, backward_h, backward_c=Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)

print('각 상태의 크기: ', lstm.shape, forward_h.shape, forward_c.shape, backward_h.shape, backward_c.shape)
state_h=Concatenate([forward_h, backward_h])#BiLSTM의 사용을 위한 concat
state_c=Concatenate([forward_c, backward_c])
attention=BahdanauAttention(64)
context_vector, attention_weights=attention(lstm, state_h)

dense1=Dense(20, activation='relu')(context_vector)
dropout=Dropout(0.5)(dense1)
output=Dense(1, activation='sigmoid')(dropout)

model=Model(inputs=sequence_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#IMDB review sentiment classification

history=model.fit(X_train, y_train, epochs=3, batch_size=256, validation_data=(X_test, y_test), verbose=1)
print('\n 테스트 정확도: ', model.evaluate(X_test, y_test)[1])
