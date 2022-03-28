"""[1. 커스터마이즈드 KoNLPy 사용하기]
한국어는 토큰화가 어렵기에 형태소 분석기를 사용하여 단어 토큰화를 진행한다. 여기서 Customized konlpy를 사용하여 사용자 사전 추가를 이용하는데,
나누면 안되는 고유명사 같은 것을 직접 add_tictionary로 품사와 함께 넣어주면 제대로 하나의 토큰으로 인식한다.
from ckonlpy.tag import Twitter

twitter=Twitter()
twitter.morphs('은경이는 사무실로 갔습니다.')
#출력->['은', '경이', '는', '사무실', '로', '갔습니다', '.'] 은경이가 분리되어버림..

twitter.add_dictionary('은경이', 'Noun')#직접 사용자 사전 추가
twitter.morphs('은경이는 사무실로 갔습니다.')
#출력->['은경이', '는', '사무실', '로', '갔습니다', '.'] 은경이를 제대로 하나의 토큰으로 인식해줌!"""
    
    #[2. 한국어 Babi 데이터셋 로드와 전처리] 훈련 데이터 : https://bit.ly/31SqtHy 테스트 데이터 : https://bit.ly/3f7rH5g
from ckonlpy.tag import Twitter
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from nltk import FreqDist
from functools import reduce
import os
import re
import matplotlib.pyplot as plt

#for load data
TRAIN_FILE=os.path.join('qa1_single-supporting-fact_train_kor.txt')
TEST_FILE=os.path.join('qa1_single-supporting-fact_test_kor.txt')

#pre-processing
def read_data(dir):
    stories, questions, answers=[], [], []
    story_temp=[]
    lines=open(dir, 'rb')

    for line in lines:
        line=line.decode('utf-8')#\b 제거
        line=line.strip()#좌우공백 제거
        idx, text=line.split(' ', 1)#id num 추출

        if int(idx)==1:#start of story
            story_temp=[]

        if '\t' in text:#QA
            question, answer, _=text.split('\t')#id_num 버림
            stories.append([x for x in story_temp if x])
            questions.append(question)
            answers.append(answer)
        else:#normal content of story
            story_temp.append(text)
    lines.close()
    return stories, questions, answers
train_data=read_data(TRAIN_FILE)#(데이터 원본 파일이 train, test로 나뉘어 있으니)
test_data=read_data(TEST_FILE)

train_stories, train_questions, train_answers=read_data(TRAIN_FILE)
test_stories, test_questions, test_answers=read_data(TEST_FILE)
print('훈련용 스토리의 개수: ', len(train_stories))#10000
print('훈련용 질문의 개수: ', len(train_questions))
print('훈련용 답변의 개수: ', len(train_answers))
print('테스트용 스토리의 개수: ', len(test_stories))#100
print('테스트용 질문의 개수: ', len(test_questions))
print('테스트용 답변의 개수: ', len(test_answers),'\n')

#vocab & max_len
twitter=Twitter()
twitter.add_dictionary('은경이', 'Noun')#Customized Konlpy
twitter.add_dictionary('경임이', 'Noun')
twitter.add_dictionary('수종이', 'Noun')
def tokenize(sent):
    #return [x.strip() for x in re.split('(\w+)?', sent) if x.strip()]#한국어의 경우 화장실로->화장실 이런식으로 형태소 분석이 되어야하는데 부적합하다.
    return twitter.morphs(sent)

def preprocess_data(train_data, test_data):
    counter=FreqDist()

    story_len=[]#save length for padding
    question_len=[]

    for stories, questions, answers in [train_data, test_data]:#train_data와 test_data는 각각의 stories, questions, answers들 을 [a,b]꼴로 합쳐서 가져온뒤 flatten으로 풀어 연산할 예정
        flatten=lambda data: reduce(lambda x, y: x+y, data)
        for story in stories:
            stories=tokenize(flatten(story))#여러 스토리(문장)를 flatten하여 모은 뒤 tokenize
            story_len.append(len(stories))
            for word in stories:
                counter[word]+=1

        for question in questions:
            questions=tokenize(question)
            questions_len=append(len(question))
            for word in question:
                counter[word]+=1

        for answer in answers:
            answer=tokenize(answer)
            for word in answer:#(answer은 label이기에 패딩할 필요가 없다)
                counter[word]+=1
                
    word2idx={word:(idx+1) for idx, (word, _) in enumerate(counter.most_common())}#빈도수기반 Vocab
    idx2word={idx:word for word, idx in word2idx.items()}

    story_max_len=np.max(story_len)
    question_max_len=np.max(question_len)

    return word2idx, idx2word, story_max_len, question_max_len
word2idx, idx2word, story_max_len, question_max_len=preprocess_data(train_data, test_data)

#vocab_size
vocab_size=len(word2idx)+1

#vectorize: 문장들을 freq vocab기반 integer encoding하여 각각의 maxlen기준 padding
def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs, Xq, Y=[], [], []
    flatten=lambda data: reduce(lambda x, y: x+y, data)

    stories, questions, answers=data
    for story, question, answer in zip(stories, questions, answers):
        xs=[word2idx[w] for w in tokenize(flatten(story))]#단어들을 integer encoding하여 리스트 append
        xq=[word2idx[w] for w in tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer])#답변은 단어가 하나

    return pad_sequences(Xs, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen), to_categorical(Y, num_classes=vocab_size)
Xstrain, Xqtrain, Ytrain=vectorize(train_data, word2idx, story_max_len, question_max_len)
Xstest, Xqtest, Ytest=vectorize(test_data, word2idx, story_max_len, question_max_len)
print('train(s,q,a) shape: ', Xstrain.shape, Xqtrain.shape, Ytrain.shape)
print('test(s,q,a) shape: ', Xstest.shape, Xqtest.shape, Ytest.shape,'\n')

    #[3. 메모리 네트워크로 QA 태스크 풀기]
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate#
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation

#hp
TRAIN_EPOCHS=120
BATCH_SIZE=32
EMBED_SIZE=50
LSTM_SIZE=64
DROPOUT_RATIO=0.3

#place holder
input_sequence=Input((story_max_len,))#story 입력
question=Input((question_max_len,))#question 입력
print('Stories: ', input_sequence)
print('Questions: ', question,'\n')

#story embedding1_(samples, story_max_len, embedding_dim) 질문 임베딩과 유사도 계산 예정.(어이 글쓴이 주석 잘못달아뒀었다구)
input_encoder_m=Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE))
input_encoder_m.add(Dropout(DROPOUT_RATIO))

#story embedding2_(samples, story_max_len, question_max_len) 
input_encoder_c=Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=question_max_len))
input_encoder_c.add(Dropout(DROPOUT_RATIO))

#question embedding_(samples, question_max_len, embedding_dim) 
question_encoder=Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=EMBED_SIZE, input_length=question_max_len))


#process of embedding
input_encoded_m=input_encoder_m(input_sequence)
input_encoded_c=input_encoder_c(input_sequence)
question_encoded=question_encoder(question)
print('Input encoded m: ', input_encoded_m)
print('Input encoded c: ', input_encoded_c)
print('Question encoded: ', question_encoded,'\n')

""" 이 부분은 내가 샘플 보면서 크기를 불필요하게 바꿔서 복잡하게 하는거같아 내가 생각하는대로 해봤는데, dot & add & concat을 모두 고려하여 크기를 맞추기 위함인 듯. 안맞네
#get similarity
match=dot([input_encoded_c, question_encoded], axes=-1, normalize=False)
match=Activation('softmax')(match)
print('Match shape: ', match,'\n')#(samples, story_max_len, embedding_dim)

#나머지 연산
response=add([match, input_encoded_m])#(samples, story_max_len, embedding_dim)
response=Permute((2,1))(response)#(samples, embedding_dim, story_max_len)
print('Response shape: ', response,'\n')

answer=concatenate([response, question_encoded])#sampled, question_max_len, embedding_dim)
print('Answer shape: ', answer)
근데 그걸 떠나서 주석이 잘못 달려있긴함. question이랑 embed된다고 주석달린게 다른거가 embed됨. 주석무시하고 코드만 일단 보면 될듯. 이거때매 더 햇갈림 이전 강좌도 그렇고"""
#유사도 계산
match=dot([input_encoded_m, question_encoded], axes=-1, normalize=False)#axes=-1때문에 정상연산. (samples, story_max_len, question_max_len) (None, 70, 5)
match=Activation('softmax')(match)
print('Match shape: ', match,'\n')#(samples, story_max_len, question_max_len) (None, 70, 5)

#ADD
response=add([match, input_encoded_c])
response=Permute((2,1))(response)#(samples, question_max_len, story_max_len) (None, 5, 70)
print('Response shape: ', response)

#Concat
answer=concatenate([response, question_encoded])#(samples, question_max_len, embedding_dim+story_max_len) (None, 5, 120)
print('Answer shape: ', answer)

answer=LSTM(LSTM_SIZE)(answer)
answer=Dropout(DROPOUT_RATIO)(answer)
answer=Dense(vocab_size)(answer)
answer=Activation('softmax')(answer)

#모델 완성
model=Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history=model.fit([Xstrain, Xqtrain],
                  Ytrain, BATCH_SIZE, TRAIN_EPOCHS,
                  validation_data=([Xstest, Xqtest], Ytest))
model.save('model.h5')

#가시화
plt.subplot(211)
plt.title('Accuracy')
plt.plot(history.history['acc'], color='g', label='train')
plt.plot(history.history['val_acc'], color='b', label='validation')
plt.legend(loc='best')

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history['loss'], color='g', label='train')
plt.plot(history.history['val_loss'], color='b', label='validation')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

ytest=np.argmax(Ytest, axis=1)
Ytest=model.predict([Xstest, Xqtest])
ytest_=np.argmax(Ytest_, axis=1)


#질문과 실제예측 비교
NUM_DISPLAY=30
print('{:18}|{:5}|{}'.format('질문', '실제값', '예측값'))
print(39*'-')

for i in range(NUM_DISPLAY):
    question=' '.join([idx2word[x] for x in Xqtest[i].tolist()])
    label=idx2word[ytest[i]]
    prediction=idx2word[ytest_[i]]
    print('{:20}: {:7} {}'.format(question, label, prediction))
