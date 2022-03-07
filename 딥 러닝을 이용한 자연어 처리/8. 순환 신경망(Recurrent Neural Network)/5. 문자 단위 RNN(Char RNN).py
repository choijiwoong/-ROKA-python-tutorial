"""RNN의 입출력단위인 벡터를 word-level에서 character-level로 변경하여 RNN을 구현할수도 있다. (임베딩은 특징추출하여 수치화하는건데 단어의 유사도를 살리기 위해 embedding layer사용)"""
    #[1. 문자 단위 RNN 언어 모델(Char RNNLM) by many-to-many: appl->pple]_embedding layer을 사용하지 않는다.
#1. 데이터에 대한 이해와 전처리
import numpy as np
import urllib.request
from tensorflow.keras.utils import to_categorical

urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", filename="11-0.txt")

f=open('11-0.txt', 'rb')
sentences=[]
for sentence in f:#realine
    sentence=sentence.strip()#cleaning
    sentence=sentence.lower()
    sentence=sentence.decode('ascii', 'ignore')
    if len(sentence)>0:
        sentences.append(sentence)
f.close()
print('sentences에서 상위 5개 원소(download test):', sentences[:5],'\n')

total_data=' '.join(sentences)
print('문자열의 길이(총 문자의 개수):', len(total_data))
print('total_data 일부 출력:',total_data[:200], '\n')

char_vocab=sorted(list(set(total_data)))#문자 집합의 생성(not 단어 집합)
vocab_size=len(char_vocab)
print('문자 집합의 크기: ', vocab_size, '\n')#56. 소문자+대문자가 52개임. 나머지는 '같은거일듯.

char_to_index=dict((char, index) for index, char in enumerate(char_vocab))#lookup table_1
print('문자 집합(char_to_index): ',char_to_index)
index_to_char={}#lookup table_2
for key, value in char_to_index.items():
    index_to_char[value]=key
print('문자 집합T(index_to_char): ', index_to_char, '\n')

#apple시퀀스를 예로들어, 입력길이가 4라면, 입출력시퀀스길이는 4이고, 입력데이터로 appl까지만 사용하니 appl을 입력하면 pple가 되도록 한다.(many_to_many)
seq_length=60#문자열로부터 다수의 샘플을 제작하기 위함.(like batch_size..?!!)
n_samples=int(np.floor((len(total_data)-1)/seq_length))
print('샘플의 수:', n_samples, '\n')

train_X=[]
train_y=[]
for i in range(n_samples):
    X_sample=total_data[i*seq_length : (i+1)*seq_length]#slicing as much as seq_length(샘플의 단위)

    X_encoded=[char_to_index[c] for c in X_sample]#integer encoding
    train_X.append(X_encoded)

    y_sample=total_data[i*seq_length+1 : (i+1)*seq_length+1]#X에서 1칸 shift된 값(appl->pple)

    y_encoded=[char_to_index[c] for c in y_sample]
    train_y.append(y_encoded)
print('X 데이터의 첫번째 샘플: ', train_X[0])#encoding: to integer
print('y 데이터의 첫번째 샘플: ', train_y[0])
print('-'*50)
print('X 데이터의 첫번째 샘플 디코딩: ', [index_to_char[i] for i in train_X[0]])#decoding: to character
print('y 데이터의 첫번째 샘플 디코딩: ', [index_to_char[i] for i in train_y[0]],'\n')

print('train_y가 train_x에서 오른쪽 한칸 shift값이 맞는지 확인하기 위한\ntrain_X[1]: \n',train_X[1], '\ntrain_y[1]:\n', train_y[1], '\n')

train_X=to_categorical(train_X)#one-hot vectorization. 단어의 경우 워드 임베딩을 위해 one-hot encoding을 적용하지 않았지만, 문자이기에 train_X도 one-hot encoding.
train_y=to_categorical(train_y)
print('train_X의 크기(shape): ', train_X.shape)
print('train_y의 크기(shape): ', train_y.shape,'\n')
"""단어는 원핫하고 문자는 원핫 안하고의 차이를 내 나름대로 정리해보면, 애초에 embedding layer가 학습가능하게 수치화하는 것인데,
단어의 경우 유사어거나 오타. 예를 들어 안녕&안농&하이 이런 것들은 뜻의 유사성이 있기에 수치화하고 학습을 통해 유사한 수치를 갖게하여
단어의 유사성을 얻을 수 있는데, 문자의 경우 그냥 문자고 의미따윈 없어서 embedding layer가 필요없고, 바로 one-hot vector로 만드는 거."""

#2. 모델 설계하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed

hidden_units=256#Hyper-parameter

model=Sequential()
model.add(LSTM(hidden_units, input_shape=(None, train_X.shape[2]), return_sequences=True)); print('LSTM의 input_shape: ', (None, train_X.shape[2]));#input_shape가 어케 들어가는 건지 뭐로 결정된건질 잘 모르겠네
model.add(LSTM(hidden_units, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))#TimeDistributed(): 각 스텝마다 cost계산하여 backpropagation & update weight

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_X, train_y, epochs=2, verbose=2)

def sentence_generation(model, length):
    ix=[np.random.randint(vocab_size)]#문자에 대한 랜덤정수    
    y_char=[index_to_char[ix[-1]]]#y(label)추출
    print(ix[-1], '번 문자', y_char[-1], '로 예측을 시작')

    X=np.zeros((1, length, vocab_size))#input sequence of LSTM

    for i in range(length):
        X[0][1][ix[-1]]=1#해당하는 입력의 one-hot vector
        print(index_to_char[ix[-1]], end="")#char을 출력
        ix=np.argmax(model.predict(X[:, :i+1, :])[0], 1)#?
        y_char.append(index_to_char[ix[-1]])
    return ('').join(y_char)

result=sentence_generation(model, 100)#문자 1개로 이어지는 문장 예측
print(result, '\n\n')
#epoch=80으로 돼어있어 해봤는데, epoch76/80에서 loss증가하네. overfitting된거같고 최종적으로 loss 0.2353으로 ㅈ같은 문장나왔는데
#뭔가 문제가 있어 보인다.. 할거없을때 너무 졸릴때 이 코드 검토해보는걸로 하고 아래 코드는 제대로 해봐야겠다. 신중하게. 흠..이대로 아래 코드 보는 의미가 있을까
#추정되는 문제: sentence를 file에서 읽을때 연속적인 method호출을 했음. 수정완료. 그대로.

    #[문자 단위 RNN(Char RNN)으로 텍스트 생성하기 by many-to-one: appl->e]
#1. 데이터에 대한 이해와 전처리
import numpy as np
from tensorflow.keras.utils import to_categorical

raw_text = '''
I get on with life as a programmer,
I like to contemplate beer.
But when I start to daydream,
My mind turns straight to wine.

Do I love wine more than beer?

I like to use words about beer.
But when I stop my talking,
My mind turns straight to wine.

I hate bugs and errors.
But I just think back to wine,
And I'm happy once again.

I like to hang out with programming and deep learning.
But when left alone,
My mind turns straight to wine.
'''

tokens=raw_text.split()#단락구분을 제거
raw_text=' '.join(tokens)#하나의 문자열로
print("raw_text: ", raw_text)

char_vocab=sorted(list(set(raw_text)))
vocab_size=len(char_vocab)
print('문자 집합: ', char_vocab)
print('문자 집합의 크기: ', vocab_size)

char_to_index=dict((char, index) for index, char in enumerate(char_vocab))
print('char_to_index: ', char_to_index,'\n')
#stude->n, tuden->t처럼 결과가 나와야함.(many_to_one)

length=11#입력 시퀀스길이10+예측대상문자1
sequences=[]
for i in range(length, len(raw_text)):
    seq=raw_text[i-length:i]
    sequences.append(seq)
print('총 훈련 샘플의 수:', len(sequences))
print('상위 10개 훈련샘플:', sequences[:10], '\n')

encoded_sequences=[]
for sequence in sequences:
    encoded_sequence=[char_to_index[char] for char in sequence]
    encoded_sequences.append(encoded_sequence)
print("인코딩된 상위 5개 sequence:", encoded_sequences[:5], '\n')

encoded_sequences=np.array(encoded_sequences)
X_data=encoded_sequences[:,:-1]
y_data=encoded_sequences[:, -1]
print("X_data 상위 5개:",X_data[:5])
print("y_data 상위 5개:",y_data[:5], '\n')

x_data_one_hot=[to_categorical(encoded, num_classes=vocab_size) for encoded in X_data]#각 문장들을 vocab에 의거 one-hot vetorization
x_data_one_hot=np.array(x_data_one_hot)
y_data_one_hot=to_categorical(y_data, num_classes=vocab_size)#y_data자체를 vocab에 의서 one-hot vectorization
print("X_data_one_hot 크기(shape): ", x_data_one_hot.shape, '\n')

#2. 모델 설계하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences

hidden_units=64

model=Sequential()#RNNLM은 embedding layer, RNN, Dense지만, char RNN이기에 embedding없이 구현한다.
model.add(LSTM(hidden_units, input_shape=(x_data_one_hot.shape[1], x_data_one_hot.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#loss와 같이 볼 정보
model.fit(x_data_one_hot, y_data_one_hot, epochs=100, verbose=2)

def sentence_generation(model, char_to_index, seq_length, seed_text, n):
    init_text=seed_text
    sentence=''

    for _ in range(n):
        encoded=[char_to_index[char] for char in seed_text]
        encoded=pad_sequences([encoded], maxlen=seq_length, padding='pre')
        encoded=to_categorical(encoded, num_classes=len(char_to_index))

        result=model.predict(encoded, verbose=0)
        result=np.argmax(result, axis=1)

        for char, index in char_to_index.items():
            if index==result:
                break

        seed_text=seed_text+char
        sentence=sentence+char
        
    sentence=init_text+sentence
    return sentence

print(sentence_generation(model, char_to_index, 10, 'I get on w', 80))#데이터의 길이를 입력 시퀀스:10+ label:1로 해둬서 seq_length인자를 10으로 하고 execution
