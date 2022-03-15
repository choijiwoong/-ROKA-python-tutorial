""" 기존의 BiLSTM에 CRF(Conditional Random Field)라는 층을 추가하여 모델을 개선시킨 LSTM+CRF모델로 Names Entity Recognition을 수행하는 내용이다.
    [1. CRF(Conditional Random Field]
LSTM을 위한 모델은 아니지만, LSTM+CRF모델로 자주 사용된다. 만약 사람과 조직만 BIO표현으로 나타내면 B-Per, I-Per, B-Org, I-ORG, O로 총 5개의 태그가 있다.
오호 기존의 CRF가 없는 LSTM모델에서는 one-hot 벡터로 label의 정확도를 추론하는데, 문장의 첫 시작단어가 I-Per 혹은 I-Org의 경우 명백하게 틀린 경우이지만
이를 다른 태그들과 동일선상에서 훈련을 시켜 찾는 비효율성이 있다. 즉 BIO의 제약사항을 위반하는 경우가 생기는 것이다.
 여기서 LSTM의 활성화함수 결과가 CRF층의 입력으로 들어가게 추가하게 되면 모델이 예측 개체명들 사이의 의존성을 고려하게 된다.
이때 CRF층은 점차적으로 BIO의 제약사항들을 학습하게 된다.
 1) 문장의 첫번째 단어에서는 I가 나오지 않는다.
 2) O-I패턴은 나오지 않는다.
 3) B-I-I패턴에서 개체명은 일관성을 유지한다. 즉, B-Per다음에 I-Org는 나오지 않는다.
즉, 양방향 LSTM은 입력 단어에 대한 양방향 문맥을 반영하며, CRF는 출력 레이블에 대한 양방향 문맥을 반영한다. 
내부의 규칙, 외부출력의 규칙을 학습시켜 Filter와 같이 사용하는 것이다.
 아래의 예시는 LSTM-CRF모델을 이전 챕터에 그대로 적용시킨 것인데, 전처리는 복습 겸 다시 입력하며 간결하게 주석 정리해보겠다."""
    #[1. 개체명 인식 데이터에 대한 이해와 전처리]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
data = pd.read_csv("ner_dataset.csv", encoding="latin1")#'Sentence #'(문장의 시작), 'Word', 'POS'(Pos), 'Tag'(BIO)열로 이루어짐

data=data.fillna(method='ffill')#null값을 이전의 값으로 fill(for sentence info of 'Sentence #')
data['Word']=data['Word'].str.lower()#lower

func=lambda temp: [(w, t) for w, t in zip(temp['Word'].values.tolist(), temp['Tag'].values.tolist())]
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]#Sentence별 Word과 Tag분리, tuple list형태로 저장.

sentences, ner_tags=[], []#Sequence Labeling Task(X_data, y_data)
for tagged_sentence in tagged_sentences:
    sentence, tag_info=zip(*tagged_sentence)
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))

plt.hist([len(s) for s in sentences], bins=50)#visualization of len before padding
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

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
y_test=to_categorical(y_test_int, num_classes=tag_size)

    #[2. BiLSTM-CRF를 이용한 개체명 인식]
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional, TimeDistributed, Embedding, Dropout#Functional API
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_crf import CRFModel
from seqeval.metrics import f1_score, classification_report

embedding_dim=128
hidden_units=64
dropout_ratio=0.3

sequence_input=Input(shape=(max_len,), dtype=tf.int32, name='sequence_input')#padding_size
model_embedding=Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(sequence_input)#padding_size->vocab_size
model_bilstm=Bidirectional(LSTM(units=hidden_units, return_sequences=True))(model_embedding)#hidden_units
model_dropout=TimeDistributed(Dropout(dropout_ratio))(model_bilstm)#dropout before dense layer
model_dense=TimeDistributed(Dense(tag_size, activation='relu'))(model_dropout)#to tag_size(stochastic one-hot vector)

base=Model(inputs=sequence_input, outputs=model_dense)#model connecting
model=CRFModel(base, tag_size)#CRF layer is added to output of LSTM. output_dim=tag_size
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), metrics='accuracy')

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc=ModelCheckpoint('bilstm_crf/cp.ckpt', monitor='val_decode_sequence_accuracy', mode='max', verbose=1, save_best_only=True, save_weights_only=True)

history=model.fit(X_train, y_train_int, batch_size=128, epochs=15, validation_split=0.1, callbacks=[mc, es])#keras-crf가 one-hot vector를 지원하지 않아 y_train이 아닌 y_train_int를 사용.

#after training, check model
model.load_weights('bilstm_crf/cp.ckpt')

i=13
y_predicted=model.predict(np.array([X_test[i]]))[0]#얘가 정수인코딩값이니
labels=np.argmax(y_test[i], -1)#원핫을 정수인코딩으로

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
    if word!=0:#PAD
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

#check performance
y_predicted=model.predict(X_test)[0]
print('\n(test)상위 2개의 y_predicted 샘플: ', y_predicted[:2])#Integer encoding값 (not one-hot encoding)

def sequences_to_tag(sequences):#one-hot용
    result=[]
    for sequence in sequences:
        word_sequence=[]
        for pred in seqeunce:
            pred_index=np.argmax(pred)#정수인코딩으로
            word_sequence.append(index_to_ner[pred_index].replace("PAD", "O"))
        result.append(word_sequence)
    return result
def sequences_to_tag_for_crf(sequences):#integer-encoding용
    result=[]
    for sequence in sequences:
        word_sequence=[]
        for pred_index in seqeunce:
            word_sequence.append(index_to_ner[pred_index].replace("PAD", "O"))
        result.append(word_sequence)
    return result
pred_tags=sequences_to_tag_for_crf(y_predicted)
test_tags=sequences_to_tag(y_test)

print('F1-score: {:.1%}'.format(f1_score(test_tags, pred_tags)))#79.1%
print(classification_report(test_tags, pred_tags))
