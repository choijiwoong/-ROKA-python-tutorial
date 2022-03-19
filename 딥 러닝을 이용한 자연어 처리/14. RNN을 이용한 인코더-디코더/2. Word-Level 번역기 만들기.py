"""태깅작업의 병렬 코퍼스와는 차이점으로 쌍의 길이가 다를 수 있다는 것에 있다."""
 #1. 데이터 로드 및 전처리
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
import unicodedata
import urllib3
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

http=urllib3.PoolManager()#도움툴
url='http://www.manythings.org/anki/fra-eng.zip'
filename='fra-eng.zip'
path=os.getcwd()
zipfilename=os.path.join(path, filename)
with http.request('GET', url, preload_content=False) as r, open(zipfilename, 'wb') as out_file:
    shutil.copyfileobj(r, out_file)
with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

num_sampled=33000#사용할 데이터 양
def to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!='Mn')#일반문장이 아니라면 NFD normalize로 악센트를 삭제한다.
def preprocess_sentence(sent):
    sent=to_ascii(sent.lower())
    sent=re.sub(r"([?.!,¿])", r" \1", sent)#단어와 구두점 사이에 공백을 추가한다. (구두점을 찾아 공백으로 capture_list1에 대하여 replace한다.)
    sent=re.sub(r'[^a-zA-Z!.?]+', r' ', sent)#글자, 3개의 특수문자 제외하고 공백으로 변환한다.
    sent=re.sub(r'\s+', ' ', sent)#여러 공백을 하나의 공백으로
    return sent
#preprocessing_test
en_sent = u"Have you had dinner?"
fr_sent = u"Avez-vous déjà diné?"
print('(test)전처리 전 영어: ', en_sent)
print('전처리 후 영어: ', preprocess_sentence(en_sent))
print('전처리 전 프랑스어: ', fr_sent)
print('전처리 후 프랑스어: ', preprocess_sentence(fr_sent),'\n')#악센트 잘 제거됨!

#전체 데이터 전처리
def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target=[], [], []

    with open('fra.txt', 'r') as lines:
        for i, line in enumerate(lines):
            src_line, tar_line, _=line.strip().split('\t')#source 데이터와 target 데이터 분리. 데이터 원본에 tab단위로 영어와 프랑스 나뉨.

            src_line=[w for w in preprocess_sentence(src_line).split()]#encoder input예정
            
            tar_line=preprocess_sentence(tar_line)
            tar_line_in=[w for w in ("<sos> "+tar_line).split()]#decoder input예정. sos추가
            tar_line_out=[w for w in (tar_line+' <eos>').split()]#decoder target예정. eos추가. 이게 맞는거같은데? 내가 기존에 잘못이해한듯. 전챕터 그림 실어둘게

            encoder_input.append(src_line)
            decoder_input.append(tar_line_in)
            decoder_target.append(tar_line_out)

            if i==num_sampled-1:
                break
    return encoder_input, decoder_input, decoder_target

sents_en_in, sents_fra_in, sents_fra_out=load_preprocessed_data()
print('인코더의 입력: ', sents_en_in[:5])
print('디코더의 입력: ', sents_fra_in[:5])
print('디코더의 레이블: ', sents_fra_out[:5])#for teacher forcing

#Use Tokenizer(단어 기준. 이전의 character-level의 경우 set으로 직접 vocab만듬)
tokenizer_en=Tokenizer(filters='', lower=False)#English전용 Tokenizer
tokenizer_en.fit_on_texts(sents_en_in)
encoder_input=tokenizer_en.texts_to_sequences(sents_en_in)
encoder_input=pad_sequences(encoder_input, padding='post')#auto

tokenizer_fra=Tokenizer(filters='', lower=False)#French전용 Tokenizer
tokenizer_fra.fit_on_texts(sents_fra_in)
tokenizer_fra.fit_on_texts(sents_fra_out)#out도 등록(eos)
decoder_input=tokenizer_fra.texts_to_sequences(sents_fra_in)#decoder_input처리
decoder_input=pad_sequences(decoder_input, padding='post')#auto

decoder_target=tokenizer_fra.texts_to_sequences(sents_fra_out)#decoder output처리
decoder_target=pad_sequences(decoder_target, padding='post')
print('인코더에 입력의 크기(shape): ', encoder_input.shape)
print('디코더의 입력의 크기(shape): ', decoder_input.shape)
print('디코더의 레이블의 크기(shape): ', decoder_target.shape)

#vocab생성
src_vocab_size=len(tokenizer_en.word_index)+1
tar_vocab_size=len(tokenizer_fra.word_index)+1
print('영어 단어 집합의 크기: ', src_vocab_size, ', 프랑스어 단어 집합의 크기: ', tar_vocab_size)

#딕셔너리 생성
src_to_index=tokenizer_en.word_index
index_to_src=tokenizer_en.index_word
tar_to_index=tokenizer_fra.word_index
index_to_tar=tokenizer_fra.index_word

#테스트 분리 전 데이터 shuffle(그냥 별거없고 기존 순서랑만 다르게 섞은거임. encoder_input, decoder_input, decoder_target의 위치는 당연히 동일)
indices=np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
print('랜덤 시퀀스: ', indices)

encoder_input=encoder_input[indices]
decoder_input=decoder_input[indices]
decoder_target=decoder_target[indices]

print('30997 encoder_input: ', encoder_input[30997])
print('30997 decoder_input: ', decoder_input[30997])
print('30997 decoder_target: ', decoder_target[30997])

#데이터 분리
n_of_val=int(num_sampled*0.1)
print('검증 데이터의 개수: ', n_of_val)

encoder_input_train=encoder_input[:-n_of_val]
decoder_input_train=decoder_input[:-n_of_val]
decoder_target_train=decoder_target[:-n_of_val]

encoder_input_test=encoder_input[-n_of_val:]
decoder_input_test=decoder_input[-n_of_val:]
decoder_target_test=decoder_target[-n_of_val:]

print('훈련 source 데이터의 크기: ', encoder_input_train.shape)#굉장히 좋은 습관
print('훈련 target 데이터의 크기: ', decoder_input_train.shape)
print('훈련 target 레이블의 크기: ', decoder_target_train.shape)
print('테스트 source 데이터의 크기: ', encoder_input_test.shape)#padding_size=8
print('테스트 target 데이터의 크기: ', decoder_input_test.shape)#padding_size=16
print('테스트 target 레이블의 크기: ', decoder_target_test.shape)#...

 #2. 기계 번역기 만들기
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model

embedding_dim=64
hidden_units=64

#인코더
encoder_inputs=Input(shape=(None,))
enc_emb=Embedding(src_vocab_size, embedding_dim)(encoder_inputs)
enc_masking=Masking(mask_value=0.0)(enc_emb)#paddding0을 연산에서 제외. 그래서 padding_size를 기존에 정하지않았던건강

encoder_lstm=LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c=encoder_lstm(enc_masking)
encoder_states=[state_h, state_c]

#디코더
decoder_inputs=Input(shape=(None,))#set
dec_emb_layer=Embedding(tar_vocab_size, hidden_units)#set
dec_emb=dec_emb_layer(decoder_inputs)#use
dec_masking=Masking(mask_value=0.0)(dec_emb)#set&use

decoder_lstm=LSTM(hidden_units, return_sequences=True, return_state=True)#set
decoder_outputs, _, _=decoder_lstm(dec_masking, initial_state=encoder_states)#use / training이기에 state사용안함. initial state setting.
decoder_dense=Dense(tar_vocab_size, activation='softmax')#set
decoder_outputs=decoder_dense(decoder_outputs)#use

model=Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])#decoder의 각 timestep은 multi-classification인데, label들이 integer encoded state이기에 sparse loss사용.

model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train, validation_data=([encoder_input_test, decoder_input_test], decoder_target_test), batch_size=128, epochs=50)

 #3. seq2seq 기계 번역기 동작시키기_인코더 마지막시점 상태, <sos>를 디코더로 보낸 뒤, 다음 <eos>까지 예측을 한다.
#인코더
encoder_model=Model(encoder_inputs, encoder_states)

#디코더
decoder_state_input_h=Input(shape=(hidden_units,))
decoder_state_input_c=Input(shape=(hidden_units,))
decoder_states_inputs=[decoder_state_input_h, decoder_state_input_c]

dec_emb2=dec_emb_layer(decoder_inputs)#재사용

decoder_outputs2, state_h2, state_c2=decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2=[state_h2, state_c2]

#모든시점 단어예측
decoder_output2=decoder_dense(decoder_outputs2)

decoder_model=Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs2]+decoder_states2)

#테스트 동작을 위한 함수(use 모델)
def decode_sequence(input_seq):
    states_value=encoder_model.predict(input_seq)#인코더 통과, get states

    #<sos>정수 생성
    target_seq=np.zeros((1,1))#굳이 shape11로 하는 이유가 뭘까앙..decoder_model의 input부의 decoder_inputs의 shape는 (None,)인데 그래서인가보다앙.
    target_seq[0,0]=tar_to_index['<sos>']

    stop_condition=False
    decoded_sentence=''
    
    while not stop_condition:
        output_tokens, h, c=decoder_model.predict([target_seq]+states_value)#use states of Encoder

        sampled_token_index=np.argmax(output_tokens[0,-1,:])#반환된 softmax vector기반 예측된 index값
        sampled_char=index_to_tar[sampled_token_index]#을 char로

        decoded_sentence+=' '+sampled_char

        if (sampled_char=='<eos>' or len(decoded_sentence)>50):
            stop_condition=True

        target_seq=np.zeros((1,1))
        target_seq[0,0]=sampled_token_index#판단한 이전의 index를 다음 for loop의 입력으로 사용하기 위함.

        states_value=[h,c]

    return decoded_sentence

#결과 확인을 위한 함수(영어와 프랑스어의 integer를 텍스트로 바꾸는 함수 for 편의)
def seq_to_src(input_seq):#원문의 정수 시퀀스를 텍스트 시퀀스로
    sentence=''
    for encoded_word in input_seq:
        if(encoded_word!=0):
            sentence=sentence+index_to_src[encoded_word]+' '
    return sentence
def seq_to_tar(input_seq):#번역문의 정수 시퀀스를 텍스트 시퀀스로
    sentence=''
    for encoded_word in input_seq:
        if(encoded_word!=0 and encoded_word!=tar_to_index['<sos>'] and encoded_word!=tar_to_index['<eos>']):
            sentence=sentence+index_to_tar[encoded_word]+' '
    return sentence

#train_data에 대한 결과(임의 index)
for seq_index in [3,50,100,300,1001]:
    input_seq=encoder_input_train[seq_index: seq_index+1]
    decoded_sentence=decode_sequence(input_seq)#모델이용, 영어->프랑스어 prediction

    print('입력 문장: ', seq_to_src(encoder_input_train[seq_index]))
    print('정답 문장: ', seq_to_tar(decoder_input_train[seq_index]))
    print('번역 문장: ', decoded_sentence[1:-5])#-5인 이유는 말그대로 sentence의 정보이기 때문에 '<eos>'를 배제하기 위함.
    print('-'*50)
#test_data에 대한 결과
for seq_index in [3, 50, 100, 300, 1001]:
    input_seq=encoder_input_test[seq_index: seq_index+1]
    decoded_sentence=decode_sequence(input_seq)

    print('입력 문장: ', seq_to_src(encoder_input_test[seq_index]))
    print('정답 문장: ', seq_to_tar(decoder_input_test[seq_index]))
    print('번역 문장: ', decoded_sentence[1:-5])
    print('-'*50)
