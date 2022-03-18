""" RNN의 many-to-one로 텍스트 분류를, many-to-many로 개체명 인식이나 품사 태깅을 해결했는데, 그 외에도
하나의 RNN을 인코더, 다른 하나를 디코더로서 두개를 연결해 사용하는 구조로 번역기, 텍스트 요약처럼 입력문장과 출력 문장의 길이가 다를 경우에 사용한다.
이번 챕터에서는 번역처럼 섬세한 자연어 처리 태스크를 기계적으로 평가가능하게하는 BLEU(Bilingual Evaluation Understudy Score)에 대하여 설명한다

    [1. 시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)]
챗봇이나 기계번역과 같은 입력시퀀스와 다른 도메인의 시퀀스를 출력하는 분야에서 사용되며, 번역기에서 대표적으로 사용된다.
seq2seq모델의 내부는 Encoder를 통하여 하나의 벡터인 context vector(인코더 RNN셀의 마지막 시점의 은닉상태)로 압축하고,
이를 디코더로 받아 번역된 단어를 한 개씩 순차적으로 출력한다.
 디코더는 기본적으로 RNNLM으로, 문장의 시작을 의미하는 심볼 <sos>가 들어가면, 문장의 끝인 <eos>가 나올 때 까지 다음단어를 예측한다.
seq2seq는 훈련과정과 테스트과정의 작동방식이 다른데, 훈련시에는 teacher forcing을 따르기에 <sos>와 <eos>를 사용한다. 반면 테스트 과정에서는 <sos>만을 입력받는다.
 이러한 과정에서 당연히 단어에 대한 임베딩을 거치게된다. RNN은 구조상 2개의 입력과 2개의 출력을 따르는데, 이 모든 입력의 누적값이 인코더의 마지막 timestep의 hidden_state를 따르기에
디코더의 입력으로 이를 사용하는 것이다.
 디코더에서는 인코더에서 나온 입력값, context vector를 받아 rnn을 돌려 예측단어를 출력하는데, 각 timestep에서의 rnn출력은 Dense, softmax를 통해
vocab확률벡터를 반환한다.

    [2. 문자 레벨 기계 번역기(Character-Level Neural Maching Translation)구현하기]
기계 번역기를 구현하기 위해서는 두 개 이상의 언어가 병렬적으로 구성된 코퍼스인 병렬 코퍼스가 필요하다."""
 #1. parallel corpus데이터에 대한 이해와 전처리. 우선 길이가 다르다.
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

http=urllib3.PoolManager()
url='http://www.manythings.org/anki/fra-eng.zip'
filename='fra-eng.zip'
path=os.getcwd()
zipfilename=os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:#읽어 zip에 쓸예정
    shutil.copyfileobj(r, out_file)
with zipfile.ZipFile(zipfilename, 'r') as zip_ref:#압축해제?
    zip_ref.extractall(path)

lines=pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')#lic은 뭔지 모르겠고 src가 eng, tar가 fre임.
del lines['lic']
print('전체 샘플의 개수: ', len(lines))#192341

lines=lines.loc[:, 'src' :'tar']
lines=lines[0:60000]#6만개만 저장

#학습을 위해선 <sos>와 <eos>를 넣어야하기에 '\t', '\n'을 기준으로 추가해보자.
lines.tar=lines.tar.apply(lambda x: '\t '+x+' \n')#target들에 \t \n추가.

#문자집합 생성
src_vocab=set()
for line in lines.src:
    for char in line:
        src_vocab.add(char)
tar_vocab=set()

for line in lines.tar:
    for char in line:
        tar_vocab.add(char)
src_vocab=sorted(list(src_vocab))
tar_vocab=sorted(list(tar_vocab))

src_vocab_size=len(src_vocab)+1
tar_vocab_size=len(tar_vocab)+1
print('source 문장의 char 집합: ', src_vocab_size)#80
print('target 문장의 char 집합: ', tar_vocab_size)#105

#문자에 인덱스 부여(vocab)생성
src_to_index=dict([(word, i+1) for i, word in enumerate(src_vocab)])#enumerate는 index, content순으로 반환한다.
tar_to_index=dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print("src_to_index: ", src_to_index)
print("tar_to_index: ", tar_to_index)

#vocab이용, 정수인코딩 수행
encoder_input=[]
for line in lines.src:
    encoded_line=[]
    for char in line:#문자별로 src_to_index통해 integer encoding
        encoded_line.append(src_to_index[char])
    encoder_input.append(encoded_line)
print('source 문장의 정수 인코딩 :',encoder_input[:5])

decoder_input=[]
for line in lines.tar:
    encoded_line=[]
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)
print('target 문장의 정수 인코딩 :',decoder_input[:5])#모든 문장 앞에 1: <sos>가 있다. 마지막엔 2가 있다<eos>
#뿐만 아니라 디코더의 예측값과 비교하기 위한 실제값이 필요한데, 이는 <sos>가 필요없기에 제거한다.(\t제거)
decoder_target=[]
for line in lines.tar:
    timestep=0#라인마다 초기화
    encoded_line=[]
    for char in line:
        if timestep>0:#라인의 시작이 아니라면 append
            encoded_line.append(tar_to_index[char])
        timestep=timestep+1#라인의 시작<sos>라면 append패스
    decoder_target.append(encoded_line)
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])

#즉, 현재 인코더의 입력값은 encoder_input, 디코더의 입력값은 '\t', '\n'을 포함한 decoder_input이며, decoder_target은 '\t'가 제거되어 '\n'만 있는상태이다.
max_src_len=max([len(line) for line in lines.src])#23
max_tar_len=max([len(line) for line in lines.tar])#76. src와 tar을 같은 크기로 패딩할 필요 없다!

encoder_input=pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input=pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target=pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')

#모든 값에 대해 one-hot encoding을 한다. ?
encoder_input=to_categorical(encoder_input)
decoder_input=to_categorical(decoder_input)
decoder_target=to_categorical(decoder_target)

 #2. 교사 강요(Teacher forcing): decoder_input을 생성한 이유로 원래는 전 RNN의 출력을 다음 RNN의 입력으로 받지만, 잘못 예측한 경우 연쇄적인 오차가 극대화되기에,
#훈련과정에서는 실제값을 다음 RNN의 입력으로 넣어 모든 셀의 훈련을 극대화시킨다.

 #3. seq2seq 기계 번역기 훈련시키기
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np

encoder_inputs=Input(shape=(None, src_vocab_size))
encoder_lstm=LSTM(units=256, return_state=True)

encoder_outputs, state_h, state_c=encoder_lstm(encoder_inputs)
encoder_states=[state_h, state_c]
