#기존의 Dor-Product Attention Score Function: score(query, key)=query^T*key
#Bahdanau Attention Score Function: score(query, key)=V^T*tanh(W1*key, W2*query)

#어텐션의 기본 아이디어는 동일하다. RNN의 모든 은닉 상태를 참고하는 것이다.
import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()#내부 production에 사용할 학습가능한 가중치 정의.
        self.W1=Dense(units)
        self.W2=Dense(units)
        self.V=Dense(1)#각 hidden_state별 Attention_score를 최종적으로 얻기 위해 Encoder's hidden_state의 개수만큼의 크기만 가진다.

    def call(self, values, query):#key==value!=query
        hidden_with_time_axis=tf.expand_dims(query, 1)#연산을 위한 dimention expand

        score=self.V(tf.nn.tanh(self.W1(values)+self.W2(hidden_with_time_axis)))#Baudanau Product

        attention_weights=tf.nn.softmax(score, axis=1)#get Attention_weight for making context_vector

        context_vector=attention_weights*values#Attention Weight와 Encoder's state_h들을 Weighted sum하여 Attention_value를 생성(context vector)
        context_vector=tf.reduce_sum(context_vector, axis=1)#기존의 values와 attention weight들을 더하여 context vector(attention_score)을 최종적으로 도출한다. 그림을 참고하자.
        #즉, values가 Encoder의 hidden_states, query는 Decoder의 t-1 hidden_state이다.

        return context_vector, attention_weights#Context Vector(Attention Value와 Attention Weights반환). 그림으로 이해하고 싶다면 dot-product부분만 W3 tanh(W1*values+W2*query)라고 바꿔서 보면 됨.

from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers
import os

sequence_input=Input(shape=(max_len,), dtype='int32')
embedded_sequences=Embedding(vocab_size, 128, input_length=max_len, mask_zero=True)(sequence_input)

lstm=Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))(embedded_sequences)
lstm, forward_h, forward_c, backward_h, backward_c=Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))(lstm)#Doubled-LSTM

state_h=Concatenate()([forward_h, backward_h])#각 상태를 concate for using it
state_c=Concatenate()([forward_c, backward_c])

attention=BahdanauAttention(64)#가중치 크기. Bidirectional을 두번지난 뒤 BiLSTM의 forward, backward state를 concate하여 attention의 입력으로 사용한다.
context_vector, attention_weights=attention(lstm ,state_h)#어텐션 메커니즘에서 은닉상태만을 사용한다.

dense1=Dense(20, activation='relu')(context_vector)
dropout=Dropout(0.5)(dense1)
output=Dense(1, activation='sigmoid')(dropout)

model=Model(inputs=sequence_input, outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #실제 구조는 

tf.keras.utils.plot_model(model, to_file='BiLSTM+BiLSTM+BahdanauAttention.png', show_shapes=True)#시각화 함 해보면 이해 마무리될듯!
