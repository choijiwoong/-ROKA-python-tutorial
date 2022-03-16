    #[1. 개체명 인식 데이터에 대한 이해와 전처리]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

 #1. 단어 기준 전처리
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
data = pd.read_csv("ner_dataset.csv", encoding="latin1")#Data Load as pd.DataFrame
print('(test)데이터의 구조를 확인할 상위 5개 샘플데이터: ', data[:5])
"""(Sentence #, Word, POS, Tag)열로 이루어져 있으며, (속한 문장에 대한 정보, 단어, 품사, BIO 개체명 인식)의 정보를 가지고 있다.
여기서 Sentence #은 해당 문장이 시작할 때 'Sentence: 1'처럼 문장의 번호를 나타내며, 문장이 시작할 때를 제외한 나머지 부분(단어)들에넌 NaN값을 띤다."""
print('총 데이터프레임 행의 개수(데이터의 개수): ', len(data))

#중복값의 유무를 확인할 것인데, 데이터 구조상 이미 Sentence #열에 NaN값이 존재하니, 다른 열(Word, Tag, POS)에 결측값이 있는지를 전반적으로 확인한다.
data.isnull().sum()#각 열별로 null의 유무 bool데이터를 더한다.
#결과는 Sentence #의 1000616개를 제외하고 다른 열의 결측값은 없는 데이터 상태이다.

#각 열에서 중복을 제외한 개수들을 확인한다.(각각 몇개의 카테고리를 띠고 있는지 확인)
print('\nsentence # 열의 중복을 제거한 값의 개수(실제 문장의 개수): ',data['Sentence #'].nunique())#47959
print('Word 열의 중복을 제거한 값의 개수(단어의 총 개수_대소문자 구분X주의): ', data.Word.nunique())#35178
print('Tag 열의 중복을 제거한 값의 개수(데이터에 사용된 BIO표현이 총 몇개인지): ', data.Tag.nunique())#17

#BIO Tag의 종류별 분포를 확인해보자.
print('\nTag 열 종류별 개수 카운트')#당연히 O가 제일 많다.
print(data.groupby('Tag').size().reset_index(name='count'))#pd.DataFrame에서 sql구문이 사용가능하다(Tag를 기준 group화하고 그 size들을 나타내는데, index이름을 count로 바꾼다)

#실제 데이터를 처리하는데에 있어서, 각 단어들 기준으로 처리하기에 문장의 정보가 단어별로 필요하다. 고로 NaN값을 그 문장의 정보로 채워주도록 한다.
data=data.fillna(method='ffill')#ffill은 NaN데이터의 앞의 정보를 가져와 채워준다.
print('\nNaN값을 문장정보로 치환 후 하위 5개 샘플 데이터: \n', data.tail())#NaN값이 성공적으로 치환되었다.

#각 열의 결측값체크, Sentence #의 NaN값의 ffill을 이용한 문장정보 삽입으로 현재 전체 데이터에서 결측값은 존재하지 않는다.
print('데이터에 Null값 유무: ', data.isnull().values.any())#False

#아까 열별 카테고리 개수를 세었는데, 단어의 경우 소문자 대문자를 구분하여 세었기에 35178개라는 수치는 정확하지 않다. 소문자화하여 다시 확인하자
data['Word']=data['Word'].str.lower()
print("Word 열의 소문자화 이후 중복제거개수(찐 단어개수): ", data.Word.nunique())#31817


#실질적인 전처리에 앞서 Sequence Labeling Task를 위해 단어와 Tag정보를 pair로 묶는다.
func=lambda temp: [(w, t) for w, t in zip(temp['Word'].values.tolist(), temp['Tag'].values.tolist())]#2. func는 해당 문장의 Word와 Tag를 한 쌍의(tuple)로 (문장단위)리스트 형태로 저장한다.
tagged_sentences=[t for t in data.groupby('Sentence #').apply(func)]#1. 각 문장별 단어 토큰들 기준으로 func를 적용하는데
print('\n문장별로 (word,tag)된 리스트(tagges_sentences)의 개수: ', len(tagged_sentences))#47959. 위에서 각 열의 개수로 구한 문장의 길이와 일치한다. well done!
print('데이터의 이해를 위한 첫번째 tagged_sentences의 샘플(문장별로 (word,Tag) 리스트형태):')
print(tagged_sentences[0],'\n')

#Sequence Labeling Task (for training_X & y)
sentences, ner_tags=[], []
for tagged_sentence in tagged_sentences:#문장별 [(word,tag), (word2,tag2), ...]
    sentence, tag_info=zip(*tagged_sentence)#unzip
    sentences.append(list(sentence))#sequence labeling task
    ner_tags.append(list(tag_info))

#당연히 문장별 길이가 일치하지 않으니, 전체 데이터 문장 별 길이 분포를 확인해야한다.(before padding)
print('샘플의 최대 길이: ', max(len(l) for l in sentences))#104
print('샘플의 평균 길이: ', sum(map(len, sentences))/len(sentences))#21.86

#tokenization before padding
src_tokenizer=Tokenizer(oov_token="OOV")# 'O'처럼 쓸모없는 데이터를 위하여 index1에 단어 'OOV'를 할당하여 전용한다.***************
tar_tokenizer=Tokenizer(lower=False)#Tag의 대문자정보는 유지하도록 한다.

src_tokenizer.fit_on_texts(sentences)#for 문장데이터
tar_tokenizer.fit_on_texts(ner_tags)#for 레이블(개체명 태깅 정보)

vocab_size=len(src_tokenizer.word_index)+1#+1하는 이유는 tokenizer의 word_index, index_word에서 index0을 쓰지 않기에 크기가 1칸 늘어난다. 즉 OOV토큰도 index1의 값을 갖는 이유이다.
tag_size=len(tar_tokenizer.word_index)+1
print('\n단어 집합의 크기: ', vocab_size)#31819(당연히 둘의 기존 크기에서 +1했으니 1개 늘어남)
print('개체명 태깅 정보 집합의 크기; ', tag_size)#18

#tokenization이 완료되었으니 Integer encoding을 수행한다. for training
X_data=src_tokenizer.texts_to_sequences(sentences)#각 문장들의 word 토큰들(리스트)
y_data=tar_tokenizer.texts_to_sequences(ner_tags)#sequence labeling task를 마쳐 X_data에 대응되는 Tag 토큰들(리스트) 즉, 위의 X_data들에 대응되는 label들이다.

#편의를 위한 변환 lookup table
word_to_index=src_tokenizer.word_index
index_to_word=src_tokenizer.index_word
index_to_ner=tar_tokenizer.index_word
ner_to_index=tar_tokenizer.word_index

#padding max_len은 기존의 길이 분포를 고려, 70으로 임의 지정
max_len=70#이 값은 단어 integer_encoded sequence들의 패딩을 위한 값이다. 즉, 단어 기준 토큰화된 각 문장들의 길이를 맞추는데 사용되는 max_len이다. 단어기준임에 유의!
X_data=pad_sequences(X_data, padding='post', maxlen=max_len)
y_data=pad_sequences(y_data, padding='post', maxlen=max_len)

#train, test split
X_train, X_test, y_train_int, y_test_int=train_test_split(X_data, y_data, test_size=.2, random_state=777)#keras-crf는 one-hot을 지원하지 않아 따로 integer encoding label을 마련한다.
y_train=to_categorical(y_train_int, num_classes=tag_size)#이 데이터들은 단어를 위한 것들이다.
y_test=to_categorical(y_test_int, num_classes=tag_size)#데이터들은 word기준 padded integer encoding들이고, one-hot vectorization을 tag_size(ner_tokenizer의 vocab크기)로 one-hot화한다.

#훈련에 앞서 각 데이터의 크기를 출력하여 구조를 다시한번 파악하고, 이해한다.
print('\n훈련 샘플 문장의 크기: ', X_train.shape)
print('훈련 샘플 레이블의 크기: ', y_train_int.shape)
print('훈련 샘플 원-핫 레이블의 크기: ', y_train.shape)
print('테스트 샘플 문장의 크기:', X_test.shape)
print('테스트 샘플 레이블의 크기: ', y_test_int.shape)
print('테스트 샘플 원-핫 레이블의 크기: ',y_test.shape,'\n\n')

 #2. 문자 기준 전처리
#char_vocab을 만들건데, 문자 단위이기에 tokenizer필요없이 set으로 만든다.
words=list(set(data['Word'].values))#데이터에사용된 Word들을 중복제거하고 리스트로 저장(for 문자 추출)
chars=set([w_i for w in words for w_i in w])#words에서 각 단어들을 꺼내고, 그 단어들에 사용된 문자들을 list에 넣는데, set화하여 중복제거, char_vocab을 생성한다.
chars=sorted(list(chars))#sort하여 list로 변환

#편의를 위한 변환 lookup table character버전
char_to_index={c: i+2 for i, c, in enumerate(chars)}#앞에 두칸을 비워두고 lookup table생성. 앞의 두칸은 특별값을 위한 것으로 OOV와 PAD를 위한다.
char_to_index['OOV']=1#'OOV'를 index1에
char_to_index['PAD']=0#'PAD'를 index0에 할당한다.

index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value]=key

#이제 이 char_to_index 그리고 index_to_char을 이용하여 word단위로 tokenized된 단어들의 정보를 문장별로 담고 있는 sentence(sequence labeling task의 산유물)를 이용하여 문자단위 parsing, paddnig한다.
max_len_char=15#하나의 단어의 길이는 최대 15를 넘지 않는다고 임의로 가정한다.

def padding_char_indice(char_indice, max_len_char):#문자 시퀀스가 들어오면 이를 max_len_char(단어의 최대 길이 15)로 padding한다.
    return pad_sequences(char_indice, max_len_char, padding='post', value=0)

def integer_coding(sentences):#tokenized 단어들이 문장별로 있는 sentences
    char_data=[]
    for ts in sentences:#각 문장별 단어 토큰 리스트들을
        word_indice=[word_to_index[t] for t in ts]#해당 문장의 단어별 인덱스를 저장 [단어1의 index, 단어2의 index, ...]
        char_indice=[[char_to_index[char] for char in t] for t in ts]#각 단어들에 속하는 문자들을 index화시킨 뒤 list로 저장한다. [[단어1의 문자1 index, 단어1의 문자2 index,...], [단어2의 문자1 index, 단어2의 문자2 index, ...] ,...]
        char_indice=padding_char_indice(char_indice, max_len_char)#문장별 단어별 char_to_index sequence들을 max_len_char(문자단위패딩=15)로 패딩

        for chars_of_token in char_indice:#각 단어별 char_to_index 토큰들에 대하여
            if len(chars_of_token)>max_len_char:#만약 길이를 max_len_char을 초과하는 데이터가 있다면
                continue#건너뛰고 패스한다.(즉 데이터 공백을 만들어버린다)****************** 원래 pad_sequences에서 maxlen을 설정하면 앞에서 자를텐데? 아예 잘린 데이터를 안쓰겠다는건가. 논리 오류가 아닐까 싶은데..
        char_data.append(char_indice)#문장별 단어->문자 tokenized된 리스트를 char_data에 추가한다.
    return char_data#모든 문장에 대하여 단어별, 문자별 index화가 완료된 리스트를 리턴한다.
X_char_data=integer_coding(sentences)

"""이부분이 논리오류같다는건데, 일단 위의 integer_coding함수에서 len(chars_of_token)>max_len_char부분에 대하여 pass하라고 하였기에
현재 X_char_data 즉 문장별로 단어별로 문자coding된 것이, 위에 word전처리에서 문장의 길이를 padding하여 단어의 개수를 70으로 맞추었는데,
X_char_data에서 pass해버려 len(char_of_token)>max_len_char즉 단어길이가 15(max_len_char)를 넘는일부 데이터의 손실로 70이 안될 수도 있으니, 이를 X_char_data에서도 문장 길이 방향으로 padding해준다."""
X_char_data=pad_sequences(X_char_data, maxlen=max_len, padding='post', value=0)

#train, test split
X_char_train, X_char_test, _, _=train_test_split(X_char_data, y_data, test_size=.2, random_state=777)#X데이터만 word기준, char기준으로 다른거지 y label은 똑같기에 굳이 하나 더 만들필요 없다.(불필요한 연산제거. word기준 split에서 이미 y_train, y_test integer encoding버전, one-hot버전 다 만듬)
X_char_train=np.array(X_char_train)#np.array화
X_char_test=np.array(X_char_test)

#훈련에 앞서 char기준 데이터와 레이블의 크기 확인
print('\n훈련 샘플 문장의 크기: ', X_train.shape)
print('훈련 샘플 char 데이터의 크기: ', X_char_train.shape)#new!
print('훈련 샘플 레이블의 크기: ', y_train.shape)
print('테스트 샘플 문장의 크기: ', X_test.shape)
print('테스트 샘플 레이블의 크기: ', y_test.shape)


    #[2. 모델에 따른 f1-score비교]
 #1. BiLSTM-CNN을 이용한 개체명 인식
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, TimeDistributed, Dropout, concatenate, Bidirectional, LSTM, Conv1D, Dense, MaxPooling1D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from seqeval.metrics import f1_score, classification_report

embedding_dim=128#단어 임베딩 용 
char_embedding_dim=64#문자 임베딩 용
dropout_ratio=0.5
hidden_units=256#단어임베딩과 문자임베딩을 concatenate하여 사용할 BiLSTM의 hidden_units
num_filters=30#문자 임베딩에 사용할 CNN의 filter(kernel)개수(feature_map의 채널수)
kernel_size=3#커널 하나의 사이즈. 1D인 char_embedding_vector에 사용할 것이기에 읽어들이 문자의 개수를 의미한다.(일반적인 경우의 CNN에서는 3x3, 5x5의 크기를 가진다)

#Word Embedding
word_ids=Input(shape=(None,), dtype='int32', name='words_input')#이쪽 shape로만 보면 맞는데...
word_embeddings=Embedding(input_dim=vocab_size, output_dim=embedding_dim)(word_ids)
"""단어기준 embedding이기에 input_dim은 src_tokenizer.word_index의 크기..+1인데 (38367, 70) vocab은 31819고..
아니 one-hot의 경우는 input_vector의 size가 vocab-size가 맞는데, 왜 Integer_encoding된 X_train에 대한 input_dim이 vocab_size인거지..?*****************
아, 크기가 아닌 Integer encoding된 데이터들이라면 그 데이터의 범위를 알아야하기에 vocab_size로 그 범위를 전달해주는건가.
맞네. keras API찾아보니 햇갈릴만한 인자이름이긴 한데 input_dim이 어휘목록의 크기를 의미하네!!!"""

#char Embedding
char_ids=Input(shape=(None, max_len_char,), name='char_input')#Embedding의 input_dim은 vocab의 size를 의미한다!!! integer encoding 범위를 알기 위해!
embed_char_out=TimeDistributed(Embedding(len(char_to_index), char_embedding_dim, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)#여기도 char_vocab크기로 했네. 내가 뭔가 잘못알고있나보다.
dropout=Dropout(dropout_ratio)(embed_char_out)

#Conv1D to char Embedding
"""합성곱 신경망은 Embedding_layer와 유사한 embedding_vector를 만들어내는데, 각 filter(kernel)을 지나면 scalar값이 나오는데 각 kernel을 통해나온
이 scalar들(Pooling을 통한)을 concatenate하여 embedding_vector를 만들어낸다. 기존의 embedding-layer와의 차이점은 parameter의 수인데, kernel의 weight들이 고정적으로 사용되기에
보다 적은 weight를 사용할 수 있다. 고로 이 embedding_vector의 크기는 kernel의 개수, 즉 num_filters가 된다."""
conv1d_out=TimeDistributed(Conv1D(kernel_size=kernel_size, filters=num_filters, padding='same', activation='tanh', strides=1))(dropout)#paddin='same'은 입출력의크기가 같음을 의미한다.(padding필요X)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)#pool_size=max_len_char. 즉 이 결과는 일반 단어들과 같이 다음 LSTM에 들어갈 예정이기에 max_len_char크기로 MaxPooling(축소_특징추출)한다.
char_embeddings=TimeDistributed(Flatten())(maxpool_out)
char_embeddings=Dropout(dropout_ratio)(char_embeddings)

#concat word, char embeddings
output=concatenate([word_embeddings, char_embeddings])#이들의 크기가 궁금하긴하네(None, None, 128), (None, None, 30) 30: num_filters와 같다.

#LSTM
output=Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout_ratio))(output)

#output_layer
output=TimeDistributed(Dense(tag_size, activation='softmax'))(output)#tag_size로 출력되게끔.

model=Model(inputs=[word_ids, char_ids], outputs=[output])
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])#y_train이 one-hot


es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('bilstm_cnn.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history=model.fit([X_train, X_char_train], y_train, batch_size=128, epochs=15, validation_split=0.1, verbose=1, callbacks=[es, mc])

#predict test
model=load_model('bilstm_cnn.h5')

i=13
y_predicted=model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])
y_predicted=np.argmax(y_predicted, axis=-1)#정수인코딩화
labels=np.argmax(y_test[i], -1)

print("{:15}|{:5}|{}".format("단어", "실제값", '예측값'))
print(35*"-")

for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
    if word!=0:#PAD의 index를 사전에 0으로 해두었다. X_test가 이미 integer encoding상태이니 바로 index!=0으로 거른다.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

#Calculate F1-score
def sequences_to_tag(sequences):
    result=[]
    for sequence in sequences:#각 문장
        word_sequences=[]#
        for pred in sequence:#각 단어
            pred_index=np.argmax(pred)#integerize
            word_sequences.append(index_to_ner[pred_index].replace("PAD", "O"))#word로 append
        result.append(word_sequence)#한 문장의 one-hot vector(단어)들을 word로 모아 append
    return result

y_predicted=model.predict([X_test, X_char_test])
pred_tags=sequences_to_tag(y_predicted)#one-hot
test_tags=sequences_to_tag(y_test)

print("F1-score: {:.1%}".format(f1_score(test_tags, pred_tags)))
print(classification_report(test_tags, pred_tags))#79.0%


 #2. BiLSTM-CNN-CRF를 이용한 개체명 인식
"""BiLSTM에 CRF를 더한게 효과적이고, BiLSTM에 문자 임베딩을 더한 것이 효과적이니 이 두가지를 합쳐본 모델."""
embedding_dim=128#Hyper parameter들은 위의 BiLSTM-CNN과 동일. 단순히 CRF층을 추가할 뿐.
char_embedding_dim=64
dropout_ratio=0.5
hidden_units=256
num_filters=30
kernel_size=3

#Word Embedding
word_ids=Input(shape=(None,), dtype='int32', name='words_input')#(?)
word_embeddings=Embedding(input_dim=vocab_size, output_dim=embedding_dim)(word_ids)

#Char Embedding
char_ids=Input(shape=(None, max_len_char,), name='char_input')#(?, max_len_char) because of padding of char
embed_char_out=TimeDistributed(Embedding(len(char_to_index), char_embedding_dim, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(char_ids)
"""Embedding()의 인자 embeddings_initializer는 default값이 'uniform'으로, 초기값 설정기라고 부른다.
keras.initializers모듈의 일부로 내장된 기능으로 keras.initializers.Initializer()을 상속받은 기능이다.
Zeros는 모든 값을 0으로, Ones는 모든 값을 1로, Constant는 특정 상수로, RandomNormal은 정규분포로, RandomUniform은 균등분포로 초기값을 생성하여 텐서를 생성한다. 이 외에도 여러개가 있다.)"""
dropout=Dropout(dropout_ratio)(embed_char_out)#Before input of Conv1D

#Conv1D to Char Embedding
conv1d_out=TimeDistributed(Conv1D(kernel_size=kernel_size, filters=num_filters, padding='same', activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(max_len_char))(conv1d_out)#pool_size=max_len_char 각 문자들이 들어가 embedding_layer(단어)를 내는 것이니 max_len_char로 pooling
char_embeddings=TimeDistributed(Flatten())(maxpool_out)
char_embeddings=Dropout(dropout_ratio)(char_embeddings)#Before concatenation(근데 Word Embedding에는 dropout안하나..)

#concatenate Word Embedding & Char Embedding after Conv1D
output=concatenate([word_embeddings, char_embeddings])

#LSTM
output=Bidirectional(LSTM(hidden_units, return_sequences=True, dropout=dropout_ratio))(output)#여기도 dropout을 하는데..

base=Model(inputs=[word_ids, char_ids], outputs=[output])
model.CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics='accuracy')

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('bilstm_cnn_crf/cp.ckpt', monitor='val_decode_sequence_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)
"""일단 val_decode_sequence_accuracy의 경우 검색으로 안나와서 대충 해독한 시퀀스 accuracy 뭔가 시퀀스를 풀어서 측정한 정확도로 직역해보고,
save_weights_only는 False일 경우 모델 전체를 저장하고(model.save(filepath)) True일 경우 가중치만 저장된다. model.save_weights(filepath)"""

history=model.fit([X_train, X_char_train], y_train_int, batch_size=128, epochs=15, validation_split=0.1, callbacks=[mc, es])#CRF는 y_train_int를 사용한다!

#predict test
model.load_weights('bilstm_cnn_crf/cp.ckpt')

i=13
y_predicted=model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])[0]#이미 integer
labels=np.argmax(y_test[i], -1)#integerize

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35*"-")

for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
    if word!=0:#Except "PAD"
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

#Calculate F1-Score
y_predicted=model.predict([X_test, X_char_test])[0]
pred_tags=sequences_to_tag_for_crf(y_predicted)#integer sequence to tag
test_tags=sequences_to_tag(y_test)

print('F1-score: {:.1}'.format(f1_score(test_tags, pred_tags)))
print(classification_report(test_tags, pred_tags))#81.0%


 #3. BiLSTM-BiLSTM-CRF를 이용한 개체명 인식
"""문자임베딩의 output을 LSTM(many-to-one)구조로 순방향 LSTM과 역방향 LSTM의 은닉 상태가 concatenate된 값이 양방향 LSTM의 출력인데, 이를 하나의 단어에 대한 단어 벡터로 간주하고,
이는 워드 임베딩을 통해 얻은 단어의 임베딩 벡터와 concatenate하여 사용해보자."""#char embedding을 단어벡터로 보는거 맞네!
embedding_dim=128
char_embedding_dim=64
dropout_ratio=0.3
hidden_units=64

#Word Embedding
word_ids=Input(batch_shape=(None, None), dtype='int32', name='word_input')
word_embeddings=Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='word_embedding')(word_ids)

#Char Embedding
char_ids=Input(batch_shape=(None, None, None), dtype='int32', name='char_input')
char_embeddings=Embedding(input_dim=(len(char_to_index)), output_dim=char_embedding_dim, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5), name='char_embedding')(char_ids)

#Char Embedding to LSTM(그냥 굉장히 간단하게 dropout도 없이 통과시키는거같은데)
char_embeddings=TimeDistributed(Bidirectional(LSTM(hidden_units)))(char_embeddings)

#concatenate Char & Word
output=concatenate([word_embeddings, char_embeddings])

#LSTM
output=Dropout(dropout_ratio)(output)#한번에 Dropout하는거랑 따로 Dropout하고 사용하는거랑 어떤 차이가 있을까? 직접 Colab돌려보고싶지만 컴파일이 여기서만 된다! 이유 무엇??!
output=Bidirectional(LSTM(units=hidden_units, return_sequences=True))(output)

#output_layer
output=TimeDistributed(Dense(tag_size, activation='relu'))(output)

base=Model(inputs=[word_ids, char_ids], outputs=[output])
model=CRFModel(base, tag_size)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics='accuracy')

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('bilstm_bilstm_crf/cp.ckpt', monitor='val_decode_sequence_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

history=model.fit([X_train, X_char_train], y_train_int, batch_size=128, epochs=15, validation_split=0.1, callbacks=[mc, es])


#predict test
model.load_weights('bilstm_bilstm_crf/cp.ckpt')

i=13#계속 같은 인덱스 사용하는 이유는 여러 모델에 대하여 같은 데이터를 넣었을때의 정확도를 비교하기 위하여
y_predicted=model.predict([np.array([X_test[i]]), np.array([X_char_test[i]])])[0]
labels=np.argmax(y_test[i], -1)

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35*'-')

for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
    if word!=0:
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

#Calculate F1-Score
y_predicted=model.predict([X_test, X_char_test])[0]
pred_tags=sequences_to_tag_for_crf(y_predicted)#for_crf와 일반 sequences_to_tag의 차이점은 one-hot vector냐 integer encoding data냐의 차이
test_tags=sequences_to_tag(y_test)

print('F1-score: {:.1%}'.format(f1_score(test_tags, pred_tags)))
print(classification_report(test_tags, pred_tags))
