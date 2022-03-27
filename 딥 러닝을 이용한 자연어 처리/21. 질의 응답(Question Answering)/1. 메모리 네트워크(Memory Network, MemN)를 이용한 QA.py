"""[1. Babi 데이터 셋]
형식: ID question[tab]answer[tab]supporting_fact ID.

1 Mary moved to the bathroom.
2 John went to the hallway.
3 Where is Mary?        bathroom        1
4 Daniel went back to the hallway.
5 Sandra moved to the garden.
6 Where is Daniel?      hallway         4
7 John moved to the office.
8 Sandra journeyed to the bathroom.
9 Where is Daniel?      hallway         4
10 Mary moved to the hallway.
11 Daniel travelled to the office.
12 Where is Daniel?     office          11
13 John went back to the garden.
14 John moved to the bedroom.
15 Where is Sandra?     bathroom        8
1 Sandra travelled to the office.
2 Sandra went to the bathroom.
3 Where is Sandra?      bathroom        2

    [2. 메모리 네트워크 구조]
스토리 문장 Value와 Key, 질문 문장 Query라고 할 때, Query와 Key의 유사도를 softmax로 정규화하여 Value에 더해 유사도 값을 더해주는 어텐션 메카니즘의 의도를 가지고 있다.
그 후 질문 문장을 유사도가 더해진 결과에 concatenate한 뒤 LSTM과 Dense를 통과시킨다."""

    #[3. Babi 데이터셋 전처리하기]
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import tarfile#tar.gz파일을 압축해제하기 위함.
from nltk import FreqDist#빈도수 저장에 특화된 딕셔너리. 알아서 0으로 초기화함.
from functools import reduce#일반화된 reduce..reduce(lambda, data)로 호출하며 data의 lambda식 수행하며 결과로 사용된 대이터 대체.
import os
import re
import matplotlib.pyplot as plt

 #1. 데이터에 대한 이해
path='babi_tasks_1-20_v1-2.tar.gz'#get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
with tarfile.open(path) as tar:
    tar.extractall()
    tar.close()

DATA_DIR='tasks_1-20_v1-2/en-10k'
TRAIN_FILE=os.path.join(DATA_DIR, 'qa1_single-supporting-fact_train.txt')
TEST_FILE=os.path.join(DATA_DIR, 'qa1_single-supporting-fact_test.txt')

#(test)훈련 상위 20개 라인 출력. 1~15까지가 한 개의 스토리이며 중간중간 질문이 있고, 숫자 1이 다시 나오면 별개의 스토리가 시작됨을 의미한다!
i=0
lines=open(TRAIN_FILE, 'rb')
for line in lines:
    line=line.decode('utf-8').strip()
    #lno, text=line.split(' ', 1)#ID와 TEXT 분리원할 시
    i=i+1
    print(line)
    if i==20:
        break

 #2. 데이터 전처리
def read_data(dir):#babi데이터 전용 읽기 함수. 스토리/질문/답변을 리턴
    stories, questions, answers=[], [], []#학습을 위해 데이터 전부 별도로 저장
    story_temp=[]
    lines=open(dir, 'rb')

    for line in lines:#babi데이터 한줄씩 리드
        line=line.decode('utf-8')#b' 제거? 바이너리 표시 제거인듯
        line=line.strip()#\n제거
        idx, text=line.split(' ', 1)#line에서 앞에 1번만 split(앞쪽 id number 분리)

        if int(idx)==1:#스토리의 시작
            story_temp=[]#하나의 스토리 단위로 담기 위함
            
        if '\t' in text:#현재 읽는 줄이 질문과 답변이 달린 줄이라면(일반 스토리에는 tab이 없다)
            question, answer, _=text.split('\t')#마지막꺼는 id임
            stories.append([x for x in story_temp if x])#지금까지의 누적 스토리를 append
            questions.append(question)
            answers.append(answer)#즉, QA가 나오는 누적 스토리 단위로 stories, questions, answers에 append
        else:#현재 읽는 줄이 일반 스토리인 경우
            story_temp.append(text)#QA가 나오면 한번에 담기 위한 temp에 append
    lines.close()
    return stories, questions, answers

train_data=read_data(TRAIN_FILE)
test_data=read_data(TEST_FILE)

train_stories, train_questions, train_answers=read_data(TRAIN_FILE)
test_stories, test_questions, test_answers=read_data(TEST_FILE)

print('훈련용 스토리의 개수: ', len(train_stories))
print('훈련용 질문의 개수: ', len(train_questions))
print('훈련용 답변의 개수: ', len(train_answers))
print('테스트용 스토리의 개수: ', len(test_stories))
print('테스트용 질문의 개수: ', len(test_questions))
print('테스트용 답변의 개수: ', len(test_answers))
print('4000번째 스토리: ', train_stories[4000])
print('4000번째 질문: ', train_questions[4000])
print('4000번째 답변: ', train_answers[4000])


def tokenize(sent):
    return [x.strip() for x in re.split('(\w+)', sent) if x and x.strip()]#데이터 있다면, (사실상)공백기준 토큰화

def preprocess_data(train_data, test_data):#vocab, max_len반환
    counter=FreqDist()#빈도수를 저장하는 딕셔너리

    flatten=lambda data: reduce(lambda x, y: x+y, data)#두 문장의 story를 하나의 문장으로 통합하는 함수(데이터가 들어오면 데이터에 관해서 reduce를 수행하는데, x,y위치에 x+y를 수행하고 대체)

    story_len=[]#각 샘플의 길이 저장
    question_len=[]

    for stories, questions, answers in [train_data, test_data]:#스토리와 질문의 길이를 리스트에 저장하며(답변제외), FreqDist이용 단어별 빈도수 계산
        for story in stories:
            stories=tokenize(flatten(story))#스토리 다 모아서
            story_len.append(len(stories))#길이 append
            for word in stories:
                counter[word]+=1#빈도수 ++
        for question in questions:
            question=tokenize(question)
            question_len.append(len(question))
            for word in question:
                counter[word]+=1
        for answer in answers:
            answer=tokenize(answer)
            for word in answer:
                counter[word]+=1

    #vocab생성
    word2idx={word: (idx+1) for idx, (word, _) in enumerate(counter.most_common())}#가장 흔한 순으로 저장(빈도수 높은 순)
    idx2word={idx: word for word, idx in word2idx.items()}

    #가장 긴 샘플의 길이
    story_max_len=np.max(story_len)
    question_max_len=np.max(question_len)

    return word2idx, idx2word, story_max_len, question_max_len#(진짜 필요한 정보 다 계산해서 한번에 리턴하넴..총총)

word2idx, idx2word, story_max_len, question_max_len=preprocess_data(train_data, test_data)
vocab_size=len(word2idx)+1
print('스토리의 최대 길이: ', story_max_len)
print('질문의 최대 길이: ', question_max_len)


def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs, Xq, Y=[], [], []
    flatten=lambda data: reduce(lambda x, y: x+y, data)

    stories, questions, answers=data
    for story, question, answer in zip(stories, questions, answers):#각 스토리, 질문, 답변 별로
        xs=[word2idx[w] for w in tokenize(flatten(story))]#단어들을 vocab사용 integer encoding
        xq=[word2idx[w] for w in tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer])#답변은 단어 하나이기에 바로 integer encoding후 append
    #story 최대 길이 패딩, question최대 길이 패딩, Y(answer)는 vocab크기만큼 원-핫 인코딩
    return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), to_categorical(Y, num_classes=len(word2idx)+1)

Xstrain, Xqtrain, Ytrain=vectorize(train_data, word2idx, story_max_len, question_max_len)
Xstest, Xqtest, Ytest=vectorize(test_data, word2idx, story_max_len, question_max_len)
print('데이터별 shape확인: ', Xstrain.shape, Xqtrain.shape, Ytrain.shape, Xstest.shape, Xqtest.shape, Ytest.shape)


    #[4. 메모리 네트워크로 QA 태스크 풀기]
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate#
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation

#하이퍼파라미터
train_epochs=120
batch_size=32
embed_size=50
lstm_size=64
dropout_rate=0.3

#입력을 담는 변수(플레이스 홀더)
input_sequence=Input((story_max_len,))
question=Input((question_max_len,))
print('Stories: ', input_sequence)
print('Questions: ', question)

#Embedding A(스토리 임베딩1)
input_encoder_m=Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=embed_size))
input_encoder_m.add(Dropout(dropout_rate))#출력: (samples, story_max_len, embedding_dim)

#Embedding C(스토리 임베딩2)
input_encoder_c=Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=question_max_len))#질문과의 유사도 계산 용이기에 출력 사이즈 동일하게
input_encoder_c.add(Dropout(dropout_rate))#출력: (samples, story_max_len, question_max_len)

#Embedding B(질문을 위한 임베딩)
question_encoder=Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=question_max_len))
question_encoder.add(Dropout(dropout_rate))#출력: (samples, question_max_len, embedding_dim). C와 B의 결과 matmul시 embed_dim나옴.

#실질적 임베딩
input_encoded_m=input_encoder_m(input_sequence)#입력값 임베딩층 통과
input_encoded_c=input_encoder_c(input_sequence)
question_encoded=question_encoder(question)
print('Input encoded m', input_encoded_m)
print('Input encoded c', input_encoded_c)
print('Question encoded', question_encoded)

#스토리 단어, 질문 단어 유사도 계산
match=dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
match=Activation('softmax')(match)
print("Match shape", match)#(sampled, story_maxlen, question_max_len)? embedding_dim아닌가

response=add([match, input_encoded_c])#(samples, story_max_len, question_max_len)
response=Permute((2,1))(response)#(samples, question_max_len, story_max_len)
print('Response shape' , response)

answer=concatenate([response, question_encoded])
print('Answer shape', answer)

answer=LSTM(lstm_size)(answer)
answer=Dropout(dropout_rate)(answer)
answer=Dense(vocab_size)(answer)
answer=Activation('softmax')(answer)

model=Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
