""" Pre-trained language model을 사용한다.

    1. ELMO(Embeddings from Language Model)
Word2vec, GloVe등의 embedding vector는 Bank로 Bank Account, River Bank의 단어가 다른데 같은 vector가 사용된다. 즉 앞의 예시와 같은 경우 제대로 반영하지 못한다.
고로 같은 표기의 다단어라도 문맥을 고려하여 임베딩하는 아이디어를 문맥을 반영한 워드 임베딩(Contextualized Word Embedding)이다.

    2. BILM(Bidirectional Language Model)의 사전 훈련
ELMO은 양쪽 방향의 언어 모델을 둘 다 학습하여 활용한다는 의미로 BILM(Bidirectional Language Model)이라고도 한다.
이는 Multi-layer을 전제로 하며, biLM의 입력이 되는 단어 벡터는 embedding layer를 사용해 얻은것이 아닌 합성곱 신경망을 이용한 벡터이다.
문자 임베딩이 subword의 정보를 참고하는 것처럼 문맥과 상관없이 dog와 doggy의 연관성을 찾아낼 수 있다.
 다만 Bidirectional RNN과 ELMO의 Bidirectional Language Model은 다른데, BRNN의 경우 순방향의 RNN hidden_state와 역방향의 RNN hidden_state를 연결하여 다음 입력으로 사용하는데,
 BILM은 두 개의 언어 모델을 별개의 모델로 보고 학습한다.

     3. BILM의 활용
임베딩 되고 있는 time step의 BILM각 층의 출력값을 가져와 concatenate하고 추가작업을 진행한다.(rnn에서 연결되는게 아니라 출력값을 연결시킨다!)
ELMO가 임베딩 벡터를 얻는 과정은 순방향, 역방향 RNN의 각 층의 출력값을 연결->각 층의 출력값 별로 가중치 부여
 ->각 층의 출력값을 모두 더함(Weighted Sum)->벡터크기결정을 위한 스칼라 매개변수를 곱한다.
결과 벡터를 ELMO representation이라고 하며, 이를 텍스트 분류, 질의응답 등의 자연어 처리에 사용할 수 있다.
ELMO는 기존의 임베딩 벡터(ex GloVe의 임베딩벡터)와 연결하여 입력으로 사용할 수 있다. biLM의 가중치를 고정시키고 scalar parameter인 gamma를 훈련과정에서 학습시킨다."""
    #ELMo 표현을 사용해서 스팸 메일 분류하기_colab
%tensorflow_version 1.x
#pip install tensorflow-hub

import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import urllib.request
import pandas as pd
import numpy as np

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)#ELMO다운 to tensorflow hub

sess = tf.Session()#default initialization on tensorflow==1.x
K.set_session(sess)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

#스팸 메일 분류 데이터 다운
urllib.request.urlretrieve("https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv", filename="spam.csv")
data = pd.read_csv('spam.csv', encoding='latin-1')
print('(test)상위 5개 데이터: ', data[:5],'\n')

#전처리 및 X, y 분리
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])#데이터의 ham과 spam을 0과1로 전처리.
y_data = list(data['v1'])#label data
X_data = list(data['v2'])#str data
print('(test)X_data상위 5개: ', X_data[:5])
print('(test)y_data상위 5개: ', y_data[:5])

#train, test 분리
print('전체 데이터의 개수: ', len(X_data))#5572
n_of_train=int(len(X_data)*0.8)
n_of_test=int(len(X_data)-n_of_train)
print("train에 사용될 개수(0.8): ", n_of_train)#4457
print('test에 사용될 개수(0.2): ', n_of_test)#1115

X_train=np.asarray(X_data[:n_of_train])
y_train=np.asarray(y_data[:n_of_train])
X_test=np.asarray(X_data[n_of_train:])
y_test=np.asarray(y_data[n_of_train:])

#tensorflow의 ELMo를 케라스버전으로 변형
def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]

#모델 설계
from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text=Input(shape=(1,), dtype=tf.string)
embedding_layer=Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)#ELMo를 이용한 Embedding layer을 거쳐 은닉층을 거친 후 1개 뉴런으로 Bi.. Cla..한다.
hidden_layer=Dense(256, activation='relu')(embedding_layer)
output_layer=Dense(1, activation='sigmoid')(hidden_layer)

model=Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=1, batch_size=60)
print('\n테스트 정확도', model.evaluate(X_test, y_test)[1])
