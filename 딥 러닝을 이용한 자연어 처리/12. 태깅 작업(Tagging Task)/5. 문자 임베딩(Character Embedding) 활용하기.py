""" 개체명 인식기의 성능향상을 위해 워드 임베딩에 문자 임베딩을 concatenate하여 사용할 수 있다."""
    #[1. 문자 임베딩(Char Embedding)을 위한 전처리]_각 문자와 매핑된 정수를 각각 Embedding layer을 거치게 하여 문자 단위 임베딩을 얻게 된다. 
#예전코드들
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
data = pd.read_csv("ner_dataset.csv", encoding="latin1")#Data Load

words=list(set(data['Word'].values))
data=data.fillna(method='ffill')#fillna by ffill
data['Word']=data['Word'].str.lower()

func=lambda temp: [(w, t) for w, t in zip(temp['Word'].values.tolist(), temp['Tag'].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]#SLT준비

sentences, ner_tags=[], []#Sequence Labeling Task(X_data, y_data)
for tagged_sentence in tagged_sentences:
    sentence, tag_info=zip(*tagged_sentence)
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))

src_tokenizer=Tokenizer(oov_token='OOV')#instantiation of Tokenizer
tar_tokenizer=Tokenizer(lower=False)
src_tokenizer.fit_on_texts(sentences)#text fitting
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size=len(src_tokenizer.word_index)+1
tag_size=len(tar_tokenizer.word_index)+1

X_data=src_tokenizer.texts_to_sequences(sentences)#Integer encoding
y_data=tar_tokenizer.texts_to_sequences(ner_tags)

word_to_index=src_tokenizer.word_index#for convenience
index_to_word=src_tokenizer.index_word
index_to_ner=tar_tokenizer.index_word
ner_to_index=tar_tokenizer.word_index

max_len=70#padding
X_data=pad_sequences(X_data, padding='post', maxlen=max_len)
y_data=pad_sequences(y_data, padding='post', maxlen=max_len)

X_train, X_test, y_train_int, y_test_int=train_test_split(X_data, y_data, test_size=.2, random_state=777)#현재 integer encoded state
y_train=to_categorical(y_train_int, num_classes=tag_size)
y_test=to_categorical(y_test_int, num_classes=tag_size)#까지 단어 단위 처리


#부터 문자 단위 처리. 전체 데이터의 모든 단어를 문자 레벨로 분해, 문자집합 생성
words=list(set(data['Word'].values))
chars=set([w_i for w in words for w_i in w])
chars=sorted(list(chars))
print('문자 집합: ', chars)

char_to_index={c: i+2 for i, c in enumerate(chars)}#for convenience
char_to_index['OOV']=1
char_to_index['pad']=0

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value]=key

#문자단위 padding
max_len_char=15

def padding_char_indice(char_indice, max_len_char):
    return pad_sequences(char_indice, maxlen=max_len_char, padding='post', value=0)

def integer_coding(sentences):
    char_data=[]
    for ts in sentences:#각 문장
        word_indice=[word_to_index[t] for t in ts]#word_to_index
        char_indice=[[char_to_index[char] for char in t] for t in ts]#char_to_index(문장의 단어들, 단어의 문자들)
        char_indice=padding_char_indice(char_indice, max_len_char)#pad

        for chars_of_token in char_indice:#각 패딩된 단어
            if len(chars_of_token)>max_len_char:
                continue#길이 초과시 건너뜀
        char_data.append(char_indice)
    return char_data
X_char_data=integer_coding(sentences)#문자 단위 정수 인코딩 결과
print('(test)기존 문장: \n', sentences[0])
print('(test)단어 단위 정수 인코딩: \n', X_data[0])
print('(test)문자 단위 정수 인코딩: \n', X_char_data[0])

#padding과정에서 길이 초과 시 무시하게끔 했기에 무시된 행도 0으로 패딩(문장 길이 방향으로 패딩)
X_char_data=pad_sequences(X_char_data, maxlen=max_len, padding='post', value=0)#이전 챕터에서 문자단위 패딩 시 사용한 기준이 max_len(70). 즉 그 크기에 맞춘 것.
print('(test)X_char_data 문장길이방향 패딩 결과: \n', X_char_data[:5],'\n')

#split
X_char_train, X_char_test, _, _=train_test_split(X_char_data, y_data, test_size=.2, random_state=777)
X_char_train=np.array(X_char_train)#list에서 nparray로
X_char_test=np.array(X_char_test)
print('(test)첫번째 훈련 데이터 출력:\n',X_train[0],'\n')#첫번째 단어의 index는 150
print('(test)150번 훈련 샘플의 단어: ', index_to_word[150])#150단어는 soldiers
print('(test)150번의 문자정수인코딩결과를 단어로: ', ' '.join([index_to_char[index] for index in X_char_train[0][0]]),'\n')#첫번째 단어의 char array를 각기 char로 변환. s o l d i e r s PAD PAD PAD PAD PAD PAD PAD

#check shape before training
print('훈련 샘플 문장의 크기: ', X_train.shape)#본래의 단어들의 integer encoding들이 들어간 데이터.
print('훈련 샘플 레이블의 크기: ', y_train.shape)#태그들의 one-hot vector
print('훈련 샘플 char 데이터의 크기: ', X_char_train.shape)#단어들의 문자단위 벡터까지 들어있는 데이터.(integer_encoding)
print('테스트 샘플 문장의 크기: ', X_test.shape)#본래 테스트(integer_encodind)
print('테스트 샘플 레이블의 크기: ', y_test.shape)#본래 테스트 레이블(one-hot)

    #[2. BiLSTM-CNN을 이용한 개체명 인식]_어렵게 생각할거 없이 기존 챕터의 train, label있는데, 단어 train까지 있다고 생각하면 됨.
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, TimeDistributed, Dropout, concatenate, Bidirectional, LSTM, Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from seqeval.metrics import f1_score, classification_report
from keras_crf import CRFModel

embedding_dim=128
char_embedding_dim=53
dropout_ratio=0.5
hidden_units=256
num_filters=30
kernel_size=3

#단어 임베딩
word_ids=Input(shape=(None,), dtype='int32', name='words_input')
word_embeddings=Embedding(input_dim=vocab_size, output_dim=embedding_dim)(word_ids)

#문자 임베딩
char_ids=Input(shape=(None, max_len_char,), name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char_to_index), char_embedding_dim, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)
dropout=Dropout(dropout_ratio)(embed_char_out)

#문자 임베딩에 Conv1D수행(단어 임베딩 인푹으로 사용 예정)
conv1d_out=TimeDistributed(Conv1D(kernel_size=kernel_size, filters=num_filters, padding='same', activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)
char_embeddings=TimeDistributed(Flatten())(maxpool_out)
char_embeddings=Dropout(dropout_ratio)(char_embeddings)

#문자임베딩과 단어임베딩의 연결
output=concatenate([word_embeddings, char_embeddings])

#LSTM
output=Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout_ratio))(output)

#출력층
output=TimeDistributed(Dense(tag_size, activation='softmax'))(output)

model=Model(inputs=[word_ids, char_ids], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])


es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('bilstm_cnn.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history=model.fit([X_train, X_char_train], y_train, batch_size=128, epochs=15, validation_split=.1, verbose=1, callbacks=[es,mc])

#test model
model=load_model('bilstm_cnn.h5')

i=13
y_predicted=model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])
y_predicted=np.argmax(y_predicted, axis=-1)
print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
for word, tag, pred in zip(X_test[i], lables, y_predicted[0]):
    if word!=0:
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

#check performance
def sequences_to_tag(sequences):#one-hot vector 리스트가 들어오면(예측된 tag정보들)
    result=[]
    for sequence in sequences:#각 원소(문장)에 대하여
        word_sequence=[]
        for pred in sequence:#각 문장의 단어에 대하여
            pred_index=np.argmax(pred)#integer encoding하고
            word_sequence.append(index_to_ner[pred_index].replace("PAD", "O"))#index이용 pag정보로 변환후 word_sequences에 저장후
        result.append(word_sequence)#문장단위로 result에 append한다.
    return result

y_predicted=model.predict([X_test, X_char_test])
pred_tags=sequences_to_tag(y_predicted)
test_tags=sequences_to_tag(y_test)
print('F1-score: ', f1_score(test_tags, pred_tags))#79%
print(classification_report(test_tags, pred_tags))
"""와..레알 끝판왕이네...정말...일단 전꺼에 그대로 써서 난잡하고 어느 포인트부터 char embedding을 해야할지 모르겠는 상태에서
char embedding을 진행하는데 전꺼에 있는 데이터를 여기저기서 가져와서 사용하고, 모델링에서는 그나마 전꺼의 모델을 아니까 익숙하긴 한데
그 속에서 char embedding이 어떻게 진행되는건지 잘 모르겠음ㅎ 근무 10분남았는데 위에꺼 최대한 이해해보고 모델부분. 데이터부분은 대충 이해가긴하는데
그 이해된게 연결이 안되는게 문제인거라 금방 이해할거같은데 모델부분은 인터넷 서칭도 좀 필요해보이고 이해하기 난해하넴 Functional API의 힘인가
 아니 뭐 char embedding은 그렇다 치는데 그 결과에 왜 Conv1D를 수행하는거지..? char embedding을 통한 결과가 단어 정보라 그 출력을
단어라고 생각해서 1D CNN으로 IMDB리뷰분석하기처럼 단어에 적용하는거같다. 일단 여기까지!!!!"""
    #[3. BiLSTM-CNN-CRF]
