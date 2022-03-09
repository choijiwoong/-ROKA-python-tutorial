"""케라스의 임베딩 층(Embedding())과 사전 훈련된 워드 임베딩을 비교할 것인데, 방대한 코퍼스를 가지기에 많이 대조된다.

    1. 케라스 임베딩 층(Keras Embedding layer)
임베딩 층의 입력시퀀스의 각 단어는 Integer Encoding되어있어야 하며, 단어->encoded num->Embedding(lookup)->Dense Vector가 되는 과정을 거친다.
케라스의 임베딩 층은 EMbedding(vocab_size, output_dim, input_length=input_length)처럼 사용하며 단어집합크기, 임베딩벡터의 차원, 입력 시퀀스의 길이를 의미한다.
Embedding()은 (number of samples, input_length)의 2D Tensor을 입력받으며, sample은 Integer Sequence(encoded)이다.
이는 (number of samples, input_length, embedding word dimentionality)의 3D Tensor을 리턴한다."""
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences=['nice great best amazing', 'stop lies', 'pitiful nerd', 'excellent work', 'supreme quality', 'bad', 'highly respectable']
y_train=[1,0,0,1,1,0,1]#Sementic Classification

tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentences)
vocab_size=len(tokenizer.word_index)+1#for padding
print('size of vocabulary: ', vocab_size)

X_encoded=tokenizer.texts_to_sequences(sentences)
print('result of integer encoding: ', X_encoded)

max_len=max(len(l) for l in X_encoded)
print('max len: ', max_len)#for decision padding size

X_train=pad_sequences(X_encoded, maxlen=max_len, padding='post')
y_train=np.array(y_train)
print('result of padding: \n', X_train)

 #훈련
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

embedding_dim=4

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))#sementic classification. output_dim=1

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)#Inter encoded data

 #[1. 사전 훈련된 GloVe를 이용한 워드 임베딩(Pre-trained Word Embeding)]
print('\n\n이전의 데이터 이용 X_train: ', X_train, ", y_train: ", y_train)
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile

urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")#pre-trained GloVe다운
zf=zipfile.ZipFile('glove.6B.zip')#unzip
zf.extractall()
zf.close()


embedding_dict=dict()
f=open('glove.6B.100d.txt', encoding='utf8')#embedding layer(lookup data)

for line in f:
    word_vector=line.split()#line을 split
    word=word_vector[0]#index=0에 위치한 단어 분리
    word_vector_arr=np.asarray(word_vector[1:], dtype='float32')#index=1~에 위치한 벡터분리
    embedding_dict[word]=word_vector_arr#사전에 추가.
f.close()

print('{}개의 Embedding vector'.format(len(embedding_dict)))
print("(test)'respectable'의 embedding vector값: ",embedding_dict['respectable'], "\n해당 벡터의 차원 수: ", len(embedding_dict['respectable']))#벡터차원수 100


embedding_matrix=np.zeros((vocab_size, 100))#X_train에 해당하는 훈련된 임베딩 값을 넣어줄 vocab,100크기의 행렬 생성
print('임베딩 행렬의 크기(shape): ', np.shape(embedding_matrix),'\n')#16,100

print('기존의 데이터의 word_index: ', tokenizer.word_index.items())
print('(test)단어 great에 매핑된 정수: ', tokenizer.word_index['great'])
print('사전 훈련된 GloVe의 great 벡터값: ', embedding_dict['great'])

for word, index in tokenizer.word_index.items():#embedding_matrix에 pre-trained data인 embedding_dict를 이용, 필요한 단어의 pre-trained vector저장
    vector_value=embedding_dict.get(word)
    if vector_value is not None:
        embedding_matrix[index]=vector_value
print("(test)embedding_matrix에서 인덱스 2의 값: ", embedding_matrix[2])
#아. embedding_dict에 pre-trained를 넣어뒀고, embedding_matrix는 X_train에 해당하는 단어들에 대해 embedding_dict에서 찾아서 딱 현재 필요한 단어벡터만 모아둔 matrix구나.


 #pre-trained된 embedding_matrix로 실질적인 훈련 없이 사용하기 위한 코드
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

output_dim=100

model=Sequential()
e=Embedding(vocab_size, output_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)#weights에 embedding_matrix{단어: embedding vector}, trainable=False(use for predict)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)


 #[2. 사전 훈련된 Word2Vec을 이용한 워드 임베딩(Pre-trained Word Embeding)]
import gensim

urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", filename="GoogleNews-vectors-negative300.bin.gz")
word2vec_model=gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)#pre-trained model(Word2Vec)
print('\n\n\nWord2Vec 모델의 크기(shape): ', word2vec_model.vectors.shape)#3000000, 300

embedding_matrix=np.zeros((vocab_size, 300))
print('임베딩 행렬의 크기(shape): ', np.shape(embedding_matrix))#16, 300(사전 훈련된 임베딩 값을 넣어줄 예정)

def get_vector(word):#get vector by pre-trained Word2Vec
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None

for word, index in tokenizer.word_index.items():#우리가 만든 X_train에 대한 integer encoded
    vector_value=get_vector(word)
    if vector_value is not None:
        embedding_matrix[index]=vector_value

print("'nice'의 임베딩 벡터값(word2vec_model)", word2vec_model['nice'])
print('단어 nice에 매핑된 정수: ', tokenizer.word_index['nice'])
print("embedding_matrix의 index1번 vector:", embedding_matrix[1])


 #pre-trained된 embedding_matrix로 실질적인 훈련 없이 사용하기 위한 코드
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, Input

model=Sequential()
model.add(Input(shape=(max_len,), dtype='int32'))
e=Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', lsos='binary_crosentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)
print('end')
