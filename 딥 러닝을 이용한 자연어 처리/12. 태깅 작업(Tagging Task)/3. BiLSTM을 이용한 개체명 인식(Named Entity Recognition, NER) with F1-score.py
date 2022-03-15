import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

    #[1. 개체명 인식 데이터에 대한 이해와 전처리]
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20RNN%20Sequence%20Labeling/dataset/ner_dataset.csv", filename="ner_dataset.csv")
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
print('(test)상위 5개 샘플 데이터:\n', data[:5])#Sentence는 문장의 구분을 위한 것
print('데이터프레임 행의 개수:', len(data))#1048575 현재는 tokenization된거라 나중에 문장을 묶으면 작아진다.
print('데이터의 Null값 유무: ', data.isnull().values.any())#True

print('\n어떤 열에 Null값이 있는지:')#Null값을 지우기 전 데이터 확인
data.isnull().sum()#다른 열에는 없지만 Sentences열에서만 1,000,616개가 나왔다. 이는 Sentence에서 문장의 구분시에만 숫자를 사용하고 문장 도중일때는 NaN으로 채워뒀기 때문. 이상없음

print('\nsentence #열의 중복을 제거한 값의 개수: ', data['Sentence #'].nunique())#전체 문장 수
print("Word열의 중복을 제거한 값의 개수: ", data.Word.nunique())#총 사용된 단어의 개수 
print('Tag열의 중복을 제거한 값의 개수: ', data.Tag.nunique())#Tag이 종류

print('\nTag 종류별 개수 카운드:')
print(data.groupby('Tag').size().reset_index(name='count'))#BIO표현에서 아무것도 의미하지 않는 O가 가장 많다.

#데이터 가공
data=data.fillna(method='ffill')#fill null all as ffill_null값의 front element로 null값을 대체
print('\n(test)null값을 ffill로 변환한 후 하위 5개 샘플 데이터:\n',data.tail())#Sentence #열의 값이 해당 문장의 정보로 바뀜(몇번째 문장인지)

print('\n데이터에 Null값이 있는지: ', data.isnull().values.any())#False! 이전 확인때 Sentence #의 규칙때문에 True였던거고 ffill방식의 fillna를 통해 매꾸니 당연히 null값이 없다.

#소문자화
data['Word']=data['Word'].str.lower()
print('\nWord열에 중복을 제거한 값의 개수: ', data.Word.nunique())#기존과 비교하여 많이 줄었다thanks to 소문자화
print('(test)Word의 소문자화 이후 상위 5개 샘플 데이터: \n', data[:5])

#단어와 개체명 태깅 정보끼리 pair로 묶기(단어에 매칭되는 tag들을 분리)
func=lambda temp: [(w,t) for w, t in zip(temp['Word'].values.tolist(), temp['Tag'].values.tolist())]#문장단위로 적용 예정
tagged_sentences=[t for t in data.groupby("Sentence #").apply(func)]#Sentence별 func를 적용하여 해당 문장단위로 pair를 한다.(sentence가 temp)
print('\nparing후 전체 tagged_sentences 샘플의 개수(문장): ', len(tagged_sentences))#당연하게도 기존의 총 문장수와 같다. paring만 했으니
print('(test)paring된 tagged_sentences의 첫번째 샘플:', tagged_sentences[0],'\n')

#Sequences Labeling Task_training을 위한 작업! X와 y를 구분한다.
sentences, ner_tags=[], []
for tagged_sentence in tagged_sentences:
    sentence, tag_info=zip(*tagged_sentence)#sentece list, tag_info list 문장별로 분리
    sentences.append(list(sentence))
    ner_tags.append(list(tag_info))
print('(test)sentences squence의 첫번째 샘플: ', sentences[0])#예측 데이터인 X에 해당.
print('(test)ner_tags squence의 첫번째 샘플: ', ner_tags[0])#예측 대상인 y에 해당.
print('(test)sentences squence의 99번째 샘플: ', sentences[98])#당연하게도 길이가 다름을 확인할 수 있다==패딩이 필요할 것이다.
print('(test)ner_tags squence의 99번째 샘플: ', ner_tags[98],'\n')

#전체 데이터 길이 분포를 확인(패딩을 위한 확인작업)
print('샘플의 최대 길이: ', max(len(l) for l in sentences))#104
print('샘플의 평균 길이: ', sum(map(len, sentences))/len(sentences), '\n')
plt.hist([len(s) for s in sentences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()#0~40의 길이를 가진다.

#패딩에 앞서 정수인코딩 진행
src_tokenizer=Tokenizer(oov_token='OOV')#데이터 구조 상 인덱스 1에는 단어 OOV를 할당하기에 해당 값 'OOV'가 oov_token임을 알려줌
tar_tokenizer=Tokenizer(lower=False)#tag정보들은 내부적으로 대문자를 유지하게 한다.
src_tokenizer.fit_on_texts(sentences)
tar_tokenizer.fit_on_texts(ner_tags)

vocab_size=len(src_tokenizer.word_index)+1#모든 단어를 사용
tag_size=len(tar_tokenizer.word_index)+1
print('단어 집합의 크기: ', vocab_size)#31819
print('개체명 태깅 정보 집합의 크기: ', tag_size)#18 다들 OOV를 고려해서 하나 늘어난건가..? 왜 위에 중복제거한 개수보다 1개씩 많지..
print('(question)단어 OOV의 인덱스: ', src_tokenizer.word_index)#확인해보니 둘 다 index 0을 사용하지 않는다! index mapping의 편리성을 위한 것 같다.
print('(question)단어 OOV의 태깅정보: ', tar_tokenizer.word_index,'\n')#참고로 OOV는 src_tokenizer.word_index[1]에 할당되어 있다.

X_data=src_tokenizer.texts_to_sequences(sentences)#Integer encoding!
y_data=tar_tokenizer.texts_to_sequences(ner_tags)
print('(test)정수 인코딩 된 X_data의 첫번째 샘플: ', X_data[0])
print('(test)정수 인코딩 된 y_data의 첫번째 샘플: ', y_data[0],'\n')

#편의를 위한 index_to_word, word_to_index
word_to_index=src_tokenizer.word_index
index_to_word=src_tokenizer.index_word
ner_to_index=tar_tokenizer.word_index
index_to_ner=tar_tokenizer.index_word
index_to_ner[0]='PAD'#index0에 태그를 할당해둔다.
print('(test)index_to_ner: ', index_to_ner,'\n')

#Integer encoded sentence를 text sequence로 바꾸는 decoding작업 테스트
decoded=[]
for index in X_data[0]:
    decoded.append(index_to_word[index])
print("(test)기존의 X_data[0]: ", sentences[0])
print('(test)After decoding to integer encoded X_data[0]: ', decoded, '\n')#동일하다!

#패딩
max_len=70
X_data=pad_sequences(X_data, padding='post', maxlen=max_len)
y_data=pad_sequences(y_data, padding='post', maxlen=max_len)

X_train, X_test, y_train_int, y_test_int=train_test_split(X_data, y_data, test_size=.2, random_state=777)#Integer encoding상태에서의 split
y_train=to_categorical(y_train_int, num_classes=tag_size)#그리고 one-hot encoding버전도 따로 저장
y_test=to_categorical(y_test_int, num_classes=tag_size)
print('훈련 샘플 문장의 크기: ', X_train.shape)#(38367, 70)
print('훈련 샘플 레이블(정수 인코딩)의 크기: ', y_train_int.shape)#(38367, 70)
print('훈련 샘플 레이블(원-핫 인코딩)의 크기: ',y_train.shape,'\n')#(38367, 70, 18)
print('테스트 샘플 문장의 크기: ', X_test.shape)#(9592, 70)
print('테스트 샘플 레이블(정수 인코딩)의 크기: ', y_test_int.shape)#(9592, 70)
print('테스트 샘플 레이블(원-핫 인코딩)의 크기: ', y_test.shape, '\n\n')#(9592, 70, 18)

    #[2. 양방향 LSTM을 이용한 개체명 인식]
"""many-to-many. padding으로 인한 0이 많은 경우 Embedding layer의 mask_zero option으로 연산에서 제외시킬 수 있다.
출력층의 TimeDistributed()는 내부적으로 각 timestep마다 weight를 update시키는 것으로, 외부적으로 LSTM을 many-to-many구조로 사용하여
LSTM의 모든 시전에 대하여 출력층을 사용할 필요가 있을 경우 사용한다.(weight를 출력층 사용 timestep마다 update)"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam

embedding_dim=128
hidden_units=256

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation=('softmax'))))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

history=model.fit(X_train, y_train, batch_size=128, epochs=6, validation_split=0.1)#0.1 for checking overfitting

#model check
i=13
y_predicted=model.predict(np.array([X_test[i]]))#X_test(integer encoded)
y_predicted=np.argmax(y_predicted, axis=-1)#one-hot to integer encoding
labels=np.argmax(y_test[i], -1)#y_test(one-hot) to integer encoding for comparing

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")
for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):#index=13 샘플의 (integer encoded)tokenized sentences, (integer encoded)tokenized labels, (integer encoded)tokenized predicted labels
    if word!=0:#PAD
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag], index_to_ner[pred]))

    #[3. F1-Score]
"""개체명 인식 시 아무것도 아니라는 의미인 'O' 태깅은 대다수의 레이블을 차지하기에 기존의 accuracy를 이용하면 이러한 불필요한 O의 prediction이 대다수를 차지하게 되어
적절하지 못한 정확도가 산출될 수 있다. 위와 같은 정확도 확인 방식의 문제는 바로 아래에서 직접 확인이 가능하다. 모든 예측값이 O인 prediction의 정확도이다."""
labels = ['B-PER', 'I-PER', 'O', 'O', 'B-MISC', 'O','O','O','O','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O','O','B-MISC','I-MISC','I-MISC','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O']
predicted=['O']*len(labels)
print('\n\n예측값(모든 prediction value가 "O"): ', predicted)

hit=0#정답 개수
for tag, pred in zip(labels, predicted):
    if tag==pred:
        hit+=1
accuracy=hit/len(labels)
print('정확도: {:.1%}'.format(accuracy),'\n')#정확도: 74.4%!!!!! 즉 어떤 개체도 찾지 못하더라도 o로만 predict하면 74%의 정확도를 얻는 것이다.

"""이러한 문제를 해결하기 위해 정확도 이외의 다른 평가방식을 도입한다. pip install seqeval
이는 정밀도(precision)과 재현률(recall)의 개념을 사용한 것인데, 정밀도는 특정 개체라고 예측한 경우 중에서 실제 특정 개체로 판명되어 예측이 일치한 비율,
재현률은 전체 특정 개체중에서 실제 특정 개체라고 정답을 맞춘 비율이다. 이 둘로부터 harmonic mean 즉 조화평균을 구한 것을 f1-score라고 한다.
f1_score=2x(정밀도x재현률)/(정밀도+재현률) 이 f1-score계산은 seqeval패키지에서 편리하게 제공한다. 이를 통해 위의 간단한 예시를 확인해보자."""
from seqeval.metrics import classificatio_report
print(classification_report([labels], [predicted]))#f1-score report에 따르면, 위의 labels와 predicted가 맞춘 것은 단 1개도 없다.

#데이터를 어느정도 정상적으로 예측한 상태에서의 f1-score를 classification_report를 이용하여 확인해보자.
labels = ['B-PER', 'I-PER', 'O', 'O', 'B-MISC', 'O','O','O','O','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O','O','B-MISC','I-MISC','I-MISC','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O']
predicted = ['B-PER', 'I-PER', 'O', 'O', 'B-MISC', 'O','O','O','O','O','O','O','O','O','O','B-PER','I-PER','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O','O']
print("데이터가 어느정도 정확하게 예측되었을 경우: \n", classification_report([labels], [predicted]))

    #[4. F1-Score로 러닝 모델 성능 측정하기]_F1-score계산을 위해서 개체명 태깅의 확률 벡터, 원-핫 벡터로부터 태깅정보 시퀀스를 반환하는 sequences_to_tag함수를 사용한다.
from seqeval.metrics import f1_score, classification_report

def sequences_to_tag(sequences):#one-hot vector 리스트가 들어오면(예측된 tag정보들)
    result=[]
    for sequence in sequences:#각 원소(문장)에 대하여
        word_sequence=[]
        for pred in sequence:#각 문장의 단어에 대하여
            pred_index=np.argmax(pred)#integer encoding하고
            word_sequence.append(index_to_ner[pred_index].replace("PAD", "O"))#index이용 pag정보로 변환후 word_sequences에 저장후
        result.append(word_sequence)#문장단위로 result에 append한다.
    return result

y_predicted=model.predict([X_test])
pred_tags=sequences_to_tag(y_predicted)#prediction tag정보로 변환
test_tags=sequences_to_tag(y_test)#test데이터 비교를 위해 tag정보로 변환

print('F1-score: {:.1%}'.format(f1_score(test_tags, pred_tags)))#f1-score연산
print(classification_report(test_tags, pred_tags))#report 출력. 평균 정확도 78%! 최저 낮은게 art고 3%정답. 최고가 gpe고 95%정답
#이어서 CRF층을 추가하여 성능을 높일 수 있다. 다음 chapter!
