"""n-gram언어 모델과 NNLM은 고정된 개수의 단어를 입력으로 받지만, timestep개념을 통해 입력의 길이를 고정하지 않을 수 있다. 이를 RNNLM이라고 한다.
이 모델을 실제 사용할때는 언어의 시퀀스가 입,출력으로 연쇄되게 추론하지만, 테스트 과정은 a b c d문장에 대해 a b c를 넣으면 b c d가 나오게 훈련되며
이들은 시퀀스로 인해 결정된 단어이다. 이러한 훈련기법을 교사 강요(teacher forcing)이라고 한다.
 모델이 t시점에 예측한 값을 t+1의 입력으로 사용하지 않고, t시점의 label(정답)을 t+1시점의 입력으로 사용한다. (잘못된 데이터 주입시 뒤의예측에 영향을 미칠 위험이 크기에)
one-hot vector가 Input_layer->Embedding layer(linear)->Hidden layer(non-linear)->Output layer을 통과하여 예측된 one-hot vector를 출력한다.
 Embedding layer는 projection layer로 lookup table을 수행하는데, 이로인한 결과벡터를 Embedding vector라고 부르기에 이를 얻는 투사층을 embedding layer라고 표현한다.
임베딩층은 단어집합크기(V)*임베딩벡터크기(M)의 행렬(e)이며, 입력단어들과 곱해진다. 이는 이전 timestep의 hidden_state과 함께 연산되어 은닉층: h=tanh(eW+hW+b)로 현재의 hidden_state를 계산한다.
이 현재의 hidden_state가 softmax를 통과하며 RNNLM의 t시점 예측벡터가 나오며, 이는 다음 단어일 확률을 나타내기에 실제 단어의 one-hot vector와 cross-entropy를 사용하면 된다.
이때의 backpropagation에서 embedding vector도 학습이 된다."""

    #[RNN을 이용한 텍스트 생성(Text Generation using RNN)]
#1. 데이터에 대한 이해와 전처리
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

text="""경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고가야 오는 말이 곱다\n"""

tokenizer=Tokenizer()#make vocabulary
tokenizer.fit_on_texts([text])#tokenizer에 text세팅
vocab_size=len(tokenizer.word_index)+1#okenizer.word_index시 index[0]을 사용하지 않고 1부터 표시해서 +1
print('단어 집합의 크기: %d'%vocab_size)
print('단어에 부여된 정수 인덱스: ', tokenizer.word_index, '\n')

sequences=list()
for line in text.split('\n'):
    encoded=tokenizer.texts_to_sequences([line])[0]#BoW기반 숫자 시퀀스로. 문장의 구분을 위해 []로 line 감싸줌.
    for i in range(1, len(encoded)):
        sequence=encoded[:i+1]#슬라이싱으로 학습에 필요한 데이터 제작(a, ab, abc, ...)
        sequences.append(sequence)
print('학습에 사용할 샘플의 개수: %d'%len(sequences))
print('학습에 사용할 전체 샘플(sequences): ', sequences)#아직 레이블로 사용할 문장별 마지막 단어가 분리되지 않음

max_len=max(len(l) for l in sequences)#문장의 최대 길이
print('샘플의 최대 길이: ', max_len)#샘플의 최대크기만큼 padding하기 위함.

sequences=pad_sequences(sequences, maxlen=max_len, padding='pre')#label데이터를 분리할 때 쉽게 분리할 수 있도록(통일된 index) padding으로 label index 통일
print('샘플의 max_len으로 padding된 sequence(전체 훈련 데이터): \n', sequences,'\n')

#레이블의 분리_a b c d에서 a b c로 d를 추론하게 하므로(교사 강요)
sequences=np.array(sequences)
X=sequences[:,:-1]#리스트의 마지막을 제외한 나머지
y=sequences[:,-1]#리스트의 마지막

print("X data: ", X)
print("y data: ", y)

y=to_categorical(y, num_classes=vocab_size)#{Y는 여기서 잘 one-hot encoding 했는데..}
print("one-hot vectorized y: ", y)#.....

#2. 모델 설계하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

#Hyper parameter인 임베딩 벡터의 차원은 10, 은닉 상태의 크기는 32이며 many-to-many를 사용한다. FC을 출력층으로 vocab만큼 배치한다.
embedding_dim=10
hidden_units=32

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))#vocab_size*embedding_dim matrix(lookup table) for learning. embedded vector
model.add(SimpleRNN(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))#vocab의 단어들의 가능성을 표시해야하기에.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])=
model.fit(X, y, epochs=200, verbose=0)

def sentence_generation(model, tokenizer, current_word, n):#모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word=current_word
    sentence=''

    for _ in range(n):
        encoded=tokenizer.texts_to_sequences([current_word])[0]#해당 단어를 integer encoding by tokenizer(이미 입력된)
        encoded=pad_sequences([encoded], maxlen=5, padding='pre')#padding. X데이터화.

        result=model.predict(encoded, verbose=0)#X데이터로 predict
        result=np.argmax(result, axis=1)#vocab 가능성 리스트중 max의 argument반환

        for word, index in tokenizer.word_index.items():
            if index==result:#해당 argument의 word와 index를 찾음.
                break

        current_word=current_word+' '+word#가능성이 크다고 찾은 단어를 current word에 concatenate(다음 loop에 사용)
        sentence=sentence+' '+word#전체 결과에 해당 단어 더하기.
    sentence=init_word+sentence#초기 워드와 찾아낸 sentence concatenate
    return sentence

print(sentence_generation(model, tokenizer, '경마장에', 4))#충분한 데이터가 없기에 정확한 횟수를 인자로 줌.
print(sentence_generation(model, tokenizer, '그의', 2))
print(sentence_generation(model, tokenizer, '가는', 5), '\n\n')

    #[2. LSTM을 이용하여 텍스트 생성하기]
#1. 데이터에 대한 이해와 전처리
import pandas as pd
import numpy as np
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

df=pd.read_csv('ArticlesApril2018.csv')#https://www.kaggle.com/aashita/nyt-comments
print("csv 데이터: ", df.head(), '\n')

print('열의 개수: ', len(df.columns))
print(df.columns)

print('headline열에 null이 있는지: ', df['headline'].isnull().values.any(), '\n')

headline=[]
headline.extend(list(df.headline.values))
print('상위 5개의 제목: ', headline[:5],'\n')#Unknown이 있는데, 이도 제거하자.

print('headline의 Unknown을 삭제 전 총 샘플의 개수', len(headline))
headline=[word for word in headline if word!='Unknown']
print('Unknown(노이즈)제거 후 샘플의 개수: ',len(headline),'\n')

print('상위 5개의 제목: ', headline[:5],'\n')

#데이터 전처리(구두점 제거, 소문자화)
def repreprocessing(raw_sentence):
    preprocessed_sentence=raw_sentence.encode('utf8').decode('ascii', 'ignore')#ignore은 옵셥으로 backslash를 삭제하는듯
    return ''.join(word for word in preprocessed_sentence if word not in punctuation).lower()#구두점 제거, 소문자화
preprocessed_headline=[repreprocessing(x) for x in headline]
print('전처리된 상위 5개 제목: ', preprocessed_headline[:5], '\n')

#단어 집합의 생성
tokenizer=Tokenizer()
tokenizer.fit_on_texts(preprocessed_headline)#모든 전처리된 제목을 tokenizer에 등록
vocab_size=len(tokenizer.word_index)+1
print('단어 집합의 크기: ', vocab_size)

#훈련 데이터의 구성
sequences=list()
for sentence in preprocessed_headline:
    encoded=tokenizer.texts_to_sequences([sentence])[0]#등록된 vocab의 index에 따라 integer coding {진짜 integer encoding됬는데 왜 손실함수로 CategoricalCrossEntropy쓰라는거야..}
    for i in range(1, len(encoded)):
        sequence=encoded[:i+1]
        sequences.append(sequence)#LSTM 전용 훈련데이터 생성(a, ab, abc, ...)
print('훈련 데이터 상위 11개: ', sequences, '\n')

#index가 어떤 단어를 의미하는지를 확인하기 위한 index_to_word 생성(lookup table)
index_to_word={}
for key, value in tokenizer.word_index.items():
    index_to_word[value]=key#거꾸로 저장
print("빈도수 상위 477번 단어(index_to_word test):", index_to_word[477], '\n')

#y데이터 분리 전 전체 샘플의 길이를 동일하게 padding.
max_len=max(len(l) for l in sequences)
print('샘플의 최대 길이:', max_len, '\n')#해당 위치로 label분리 예정

sequences=pad_sequences(sequences, maxlen=max_len, padding='pre')#label이 뒤에있기에 pre padding
print('padding된 상위 3개 데이터: ', sequences[:3])

#padding된 데이터로부터 label 분리
sequences=np.array(sequences)
X=sequences[:, :-1]#마지막 -1
y=sequences[:, -1]#마지막
print('label 분리된 X 상위 데이터 3개(label이 분리되었기에 당연히 샘플의 길이가 -1됨): ', X[:3], '\n')
print('label 분리된 y 상위 데이터 3개:', y[:3], '\n\n')

#2. 모델 설계하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

embedding_dim=10
hiddden_units=128

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='SparseCategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])#??? 아 야 위에 예시랑 다르게 y one-hot encoding빠졌다 수정하자.
model.fit(X, y, epochs=200, verbose=2)

def sentence_generation(model, tokenizer, current_word, n):
    init_word=current_word
    sentence=''

    for _ in range(n):
        encoded=tokenizer.texts_to_sequences([current_word])[0]
        encoded=pad_sequences([encoded], maxlen=max_len-1, padding='pre')

        result=model.predict(encoded, verbose=0)
        result=np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items():
            if index==result:
                break

        current_word=current_word+' '+word
        sentence=sentence+' '+word
    sentence=init_word+sentence
    return sentence

#print("i를 통해 생성된 문장(10개 단어): ", sentence_generation(model, tokenizer, 'i' 10))
#print("how를 통해 생성된 문장(10개 단어): ", sentence_generation(model, tokenizer, 'how', 10))
print(sentence_generation(model, tokenizer, 'i', 10))
print(sentence_generation(model, tokenizer, 'how', 10))
