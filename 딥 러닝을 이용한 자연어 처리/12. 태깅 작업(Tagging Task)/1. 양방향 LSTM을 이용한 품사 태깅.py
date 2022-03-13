""" 각 단어가 어떤 유형에 속해있는지를 알아내는 작업으로, 각 단어의 유형이 사람, 장소, 단체 등 어떤 유형인지를 알아내는 개체명 인식(Named Entity Recognition)과
품사가 명사, 동사, 형용사 인지를 알아내는 품사 태깅(Part-of-Speech Tagging)이 있다. 개체명 인식기와 품사 태거는 둘 다 RNN의 many-to-many작업이며
Bidirectional RNN을 사용한다.
 태깅 작업은 텍스트 분류와 같이 Supervises Learning에 해당하며, X(단어)와 y데이터(태깅정보)가 쌍(pair)을 이루는 병렬구조를 가진다는 것이 특징이다.
참고로 입력 시퀀스X=[...]에 대해 쌍을 이루는 레이블 시퀀스y=[...]를 각각 부여하는 작업을 시퀀스 레이블링 작업(Sequence Labeling Task)라고 한다. 예시로 태깅작업이 있다.
 RNN의 은닉층은 return_sequences인자로 모든 시점에 대하여 은닉상태의 값을 출력할 수도, 마지막 시점에 대해서만 은닉상태의 값을 출력할 수 있다.
태깅작업의 경우 many-to-many이기에 return_sequences=True로 모든 은닉 상태의 값을 보낸다. 태깅 작업이 RNN구조에서 어떻게 진행되는지는 사진을 참고하자."""
 #1. 품사 태깅 데이터에 대한 이해와 전처리
import nltk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

tagged_sentences=nltk.corpus.treebank.tagged_sents()#영어 코퍼스에 토큰화와 품사 태깅 전처리를 한 문장 데이터 다운(훈련에 사용예정)
print('품사 태깅이 된 문장의 개수: ', len(tagged_sentences))
print('(test)첫번째 샘플: ', tagged_sentences[0],'\n')#품사 태깅 전처리를 확인할 수 있다. 여기서 Sequence Labeling Task를 통해 단어와 품사정보를 분리해야한다.

#Sequence Labeling Task
sentences, pos_tags=[], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info=zip(*tagged_sentence)#*의 이유: zip(*)은 zipped 요소들을 unzip해준다. 즉, word:tag로 zip된게 unzip되며 sentence, tag_info로 분리된 것이다.
    sentences.append(list(sentence))#list의 이유: 이미 tokenization되어 있어 그대로 넣으면 문장간의 구분이 사라진다.
    pos_tags.append(list(tag_info))
print('(test)첫번째 샘플의 단어: ', sentences[0])
print('(test)첫번째 샘플의 품사: ', pos_tags[0])
print('(test)9번째 샘플의 단어: ', sentences[8])#안그래도 알고있었지만 샘플의 길이들이 제각각임을 알 수 있다!
print('(test)9번째 샘플의 품사: ', pos_tags[8],'\n')

#length check
print('샘플의 최대 길이: ', max(len(l) for l in sentences))
print('샘플의 평균 길이: ', sum(len(l) for l in sentences)/len(sentences),'\n')#or sum(map(len, sentences))/len(sentences)
plt.hist([len(s) for s in sentences], bins=50)#y값만 입력해주면 된다
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()#대부분 150이내, 0~50길이

#sample에 해당하는 tokenizer를 반환한다(문장 데이터와 품사 태깅 정보에 다른 tokenizer을 사용하기 위함)
def tokenize(samples):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(samples)
    return tokenizer
src_tokenizer=tokenize(sentences)#문장
tar_tokenizer=tokenize(pos_tags)#품사

#각각의 vocab크기를 확인한다.
vocab_size=len(src_tokenizer.word_index)+1
tag_size=len(tar_tokenizer.word_index)+1
print('단어 집합의 크기: ', vocab_size)#11388
print('태깅 정보 집합의 크기: ', tag_size,'\n')#47

#Integer encoding
X_train=src_tokenizer.texts_to_sequences(sentences)
y_train=tar_tokenizer.texts_to_sequences(pos_tags)
print('(test)정수인코딩된 3번째 샘플 문장: ', X_train[2])
print('(test)정수인코딩된 3번째 샘플 품사: ', y_train[2],'\n')

#padding
max_len=150
X_train=pad_sequences(X_train, padding='post', maxlen=max_len)
y_train=pad_sequences(y_train, padding='post', maxlen=max_len)

#split
X_train, X_test, y_train, y_test=train_test_split(X_train, y_train, test_size=.2, random_state=777)
print('훈련 샘플 문장의 크기: ', X_train.shape)
print('훈련 샘플 레이블의 크기: ', y_train.shape)
print('테스트 샘플 문장의 크기: ', X_test.shape)
print('테스트 샘플 레이블의 크기: ', y_test.shape,'\n')


 #2. 양방향 LSTM(Bi-directional LSTM)으로 Pos Tagger만들기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam

embedding_dim=128
hidden_units=128

model=Sequential()#mask_zero는 모델에게 padding을 통해 데이터의 일부가 실제로 채워져있다는 사실을 알리는 역활로 그 외에 keras.layers.Masking레이어를 도입할 수도 있다.
model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))#왜 max_len이 아닌 vocab_size냐?: 원래 vocabsize, output_dim, input_length순으로 받는다. vocab을 참고하여 embedding하기 위함인 것 같다. max_len을 넣으려거든 input_length에 넣으면 되지만 이미 다 padding되어 같아 유추가능하다.
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))#각 step마다 cost를 계산하여 오류를 전파하며 update하라.

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])#이쯤되니 햇갈리네..이건 Multi-classification이 아닌건가..? 
model.fit(X_train, y_train, batch_size=128, epochs=7, validation_data=(X_test, y_test))#Label Encoding은 순서의미O & 고유값 개수가 많을때, One-hot Encoding은 순서X & 고유값 개수가 적을때

#확인
index_to_word=src_tokenizer.index_word
index_to_tag=tar_tokenizer.index_word

i=10#확인하고픈 인덱스
y_predicted=model.predict(np.array([X_test[i]]))
y_predicted=np.argmax(y_predicted, axis=-1)#y_predicted된 것들 중 가장 확률이 높은 tag의 인덱스
print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
for word, tag, pred in zip(X_test[i], y_test[i], y_predicted[0]):
    if word!=0:#0은 PAD값임.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_tag[tag].upper(), index_to_tag[pred].upper()))
