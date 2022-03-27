""" seq2seq를 이용하여 구현이 가능하며, attention mechanism을 적용한다.
    [1. 텍스트 요약(Text Summarization)]
크게 추출적 요약(Extractive Summarization)과 추상적 요약(Abstractive Summarization)으로 나뉜다.

 1. Extractive Summarization
핵심 문장, 단어구를 추출하여 만드는 것으로, 전부 원문의 문장을 사용한다. 대표적인 알고리즘으로 TextRank가 있다.

 2. Abstractive Summarization
원문에 없더라도 핵심 문*맥을 반영한 새로운 문장을 작성하는 것으로 대표적으로 seq2seq를 사용한다. 단점은 ANN기반 지도학습이기에 data & label구하기가 좀 그르타

    [2. 아마존 리뷰 데이터에 대한 이해]"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import urllib.request
np.random.seed(seed=0)

 #1. 데이터 로드하기(https://www.kaggle.com/snap/amazon-fine-food-reviews)
data=pd.read_csv("archive/Reviews.csv", nrows=100000)#10만개의 행으로 제한. 10개의 열이 존재하는 데이터인데 Text와 Summary열을 사용을 사용한다.
print('전체 리뷰 개수: ', len(data))#100000

data=data[['Text', 'Summary']]#필요한 데이터만 추출
print('랜덤으로 10개 샘플 출력:\n', data.sample(10))

 #2. 데이터 정제하기
print('Text열 유일한 샘플의 수: ', data['Text'].nunique())#88426
print('Summary열 유일한 샘플의 수: ', data['Summary'].nunique())#72348(하지만 요약문이기에 짧은 리뷰의 경우 같을 수 있다 가정하고 'Text'기준으로만 중복제거
data.drop_duplicates(subset=['Text'], inplace=True)
print('Text기준 중복 제거 후 전체 샘플 수: ', len(data))

print('결측값 확인(null): ', data.isnull().sum())#Summary에 1개 존재 확인
data.dropna(axis=0, inplace=True)#결측값 제거
print('결측값 제거 후 전체 샘플의 개수: ', len(data))#88425
#까지 불필요한 샘플의 수 제거. 이제 내부 처리.


contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
#약어를 풀기 위한 사전(https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python)

stop_words=set(stopwords.words('english'))
print('불용어 개수: ', len(stop_words))

def preprocess_sentence(sentence, remove_stopwords=True):#전처리된 tokenized sentence 반환
    sentence=sentence.lower()
    sentence=BeautifulSoup(sentence, 'lxml').text#html태그 제거
    sentence=re.sub(r'\([^)]*\)', '', sentence)#괄호 문자열 제거(어차피 중요X데이터 일테니. 이 괄호 속 문장처럼)
    sentence=' '.join([contractions[t] if t in contractions else t for t in sentence.split(' ')])#위에 미리 만들어둔 약어사전 이용. 약어 정규화
    sentence=re.sub(r"'s\b", "", sentence)#소유격 제거
    sentence=re.sub('[m]{2,}', 'mm', sentence)#ummmmmmmm...->umm...으로 mm처리
    
    if remove_stopwords:#구분하는 이유는 Summary에서 불용어 제거하면 너무 없어질거 같아 Text와 Summary별도 전처리를 위함.
        tokens=' '.join(word for word in sentence.split() if not word in stop_words if len(word)>1)
    else:
        tokens=' '.join(word for word in sentence.split() if len(word)>1)
        
    return tokens
#function test
temp_text='Everythins I bought was great, infact I ordered twice and the third ordered was<br />for my mother and father.'
temp_summary='Great way to start (or finish) the day!!!'
print("preprocess_sentence(temp_text): ", preprocess_sentence(temp_text))
print('preprocess_sentence(temp_summary, 0): ', preprocess_sentence(temp_summary, 0))#Well done!

clean_text=[]#preprocess_sentence 실제 적용
for s in data['Text']:
    clean_text.append(preprocess_sentence(s))

clean_summary=[]
for s in data['Summary']:
    clean_summary.append(preprocess_sentence(s,0))#불용어처리 False
data['Text']=clean_text
data['Summary']=clean_summary

#전처리 과정 중 Null값 생겼을 가능성 존재.
data.replace('', np.nan, inplace=True)
print('preprocess_sentence 이후 Null값 존재유무 확인: ', data.isnull().sum())#Sumamry에서 70개 샘플 Null.
data.dropna(axis=0, inplace=True)
print('전처리&결측값 제거 이후 전체 샘플 수: ', len(data))

#Text와 Summary의 길이 분포 확인 for padding
text_len=[len(s.split()) for s in data['Text']]#문장별 단어개수
summary_len=[len(s.split()) for s in data['Summary']]

print('텍스트 최소 길이: ', np.min(text_len))
print('텍스트 최대 길이: ', np.max(text_len))
print('텍스트 평균 길이: ', np.mean(text_len))
print('요약 최소 길이: ' ,np.min(summary_len))
print('요약 최대 길이: ', np.max(summary_len))
print('요약 평균 길이: ', np.mean(summary_len))

plt.subplot(1,2,1)#summary와 text길이 비교
plt.boxplot(summary_len)
plt.title('Summary')
plt.subplot(1,2,2)
plt.boxplot(text_len)
plt.title('Text')
plt.tight_layout()
plt.show()

plt.title('Summary')#summary
plt.hist(summary_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()#10 컷해도 될거같은데

plt.title('Text')#text
plt.hist(text_len, bins=40)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()#150컷해도 될거같고

text_max_len=50#히익
summary_max_len=8#하긴 최대한 줄여야 불필요한 패딩 사라지고 정확도도 높아지지..
def below_threshold_len(max_len, nested_list):#적절한 패딩인지 확인
    cnt=0
    for s in nested_list:
        if(len(s.split())<=max_len):
            cnt=cnt+1
    print('전체 샘플 중 길이가 %s이하인 샘플의 비율: %s'%(max_len, (cnt/len(nested_list))))
below_threshold_len(text_max_len, data['Text'])#77% 즉 23%샘플이 text_max_len보다 크다(23%의 정보손실 예정)
below_threshold_len(summary_max_len, data['Summary'])#94%

#text_max_len과 summary_max_len보다 큰 데이터 삭제(for 정확성..끄흡 살을 내주고 뼈를 취한다..)
data=data[data['Text'].apply(lambda x: len(x.split()) <= text_max_len)]
data=data[data['Summary'].apply(lambda x: len(x.split()) <= summary_max_len)]
print('max_len초과 데이터 삭제 후 전체 샘플 수: ', len(data))#65818

#seq2seq훈련을 위한 시작 토큰과 종료 토큰 추가
data['decoder_input']=data['Summary'].apply(lambda x: 'sostoken '+x)#각 문장 앞에 sostoken추가(decoder_input)
data['decoder_target']=data['Summary'].apply(lambda x: x+' eostoken')#각 문장 뒤에 eostoken추가(decoder_output)

encoder_input=np.array(data['Text'])
decoder_input=np.array(data['decoder_input'])#지도
decoder_target=np.array(data['decoder_target'])#for cal loss

 #3. 데이터의 분리
indices=np.arange(encoder_input.shape[0])
np.random.shuffle(indices)

encoder_input=encoder_input[indices]#shuffle
decoder_input=decoder_input[indices]
decoder_target=decoder_target[indices]

n_of_val=int(len(encoder_input)*0.2)#0.2비율로 Test, Train분리
print('test데이터의 수: ', n_of_val)#13163

encoder_input_train=encoder_input[:-n_of_val]#어차피 shuffle되어있어서 개수기준 나누어도 상관X
decoder_input_train=decoder_input[:-n_of_val]
decoder_target_train=decoder_target[:-n_of_val]

encoder_input_test=encoder_input[-n_of_val:]
decoder_input_test=decoder_input[-n_of_val:]
decoder_target_test=decoder_target[-n_of_val:]

print('훈련 데이터의 개수: ', len(encoder_input_train))#52655
print('훈련 레이블의 개수: ', len(decoder_input_train))#52655
print('테스트 데이터의 개수: ', len(encoder_input_test))#13163
print('테스트 레이블의 개수: ', len(decoder_input_test))#13163

 #4. 정수 인코딩
src_tokenizer=Tokenizer()
src_tokenizer.fit_on_texts(encoder_input_train)

threshold=7
#빈도수가 낮은 단어들을 제거하기 전에 적절한지 확인
total_cnt=len(src_tokenizer.word_index)
rare_cnt=0
total_freq=0
rare_freq=0

for key, value in src_tokenizer.word_counts.items():#단어별 빈도수
    total_freq=total_freq+value

    if (value<threshold):
        rare_cnt=rare_cnt+1
        rare_freq=rare_freq+1
print('단어 집합의 크기: ', total_cnt)#32031(총 vocab)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold-1, rare_cnt))#23779(희귀 수)
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기: %s'%(total_cnt-rare_cnt))#8252(빼면 몇개)
print('단어 집합에서 희귀 단어의 비율: ', (rare_cnt/total_cnt)*100)#74(개수비율은 몇)
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율: ', (rare_freq/total_freq)*100)#3(빈도비율은 몇) 적합!

src_vocab=8000#정확히는 8252이지만 크게 상관X
src_tokenizer=Tokenizer(num_words=src_vocab)
src_tokenizer.fit_on_texts(encoder_input_train)#찾아낸 희귀단어 커팅 기준으로 다시 Tokenizer data fitting

encoder_input_train=src_tokenizer.texts_to_sequences(encoder_input_train)#이 Tokenizer 기준으로 정수 인코딩
encoder_input_test=src_tokenizer.texts_to_sequences(encoder_input_test)


#레이블에 대해서도 동일한 작업 수행
tar_tokenizer=Tokenizer()
tar_tokenizer.fit_on_texts(decoder_input_train)

threshold=6
total_cnt=len(tar_tokenizer.word_index)
rare_cnt=0
total_freq=0
rare_freq=0

for key, value in tar_tokenizer.word_counts.items():
    total_freq=total_freq+value

    if (value<threshold):
        rare_cnt=rare_cnt+1
        rare_freq=rare_freq+value
print('단어 집합의 크기: ', total_cnt)#10510
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold-1, rare_cnt))#8128
print('단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기: ', total_cnt-rare_cnt)#2382
print('단어 집합에서 희귀 단어의 비율: ', rare_cnt/total_cnt*100)#77%
print('전체 등장 빈도에서 희귀 단어 등장 빈도 비율: ', rare_freq/total_freq*100)#5.8% (적합)

tar_vocab=2000
tar_tokenizer=Tokenizer(num_words=tar_vocab)
tar_tokenizer.fit_on_texts(decoder_input_train)
tar_tokenizer.fit_on_texts(decoder_target_train)

decoder_input_train=tar_tokenizer.texts_to_sequences(decoder_input_train)
decoder_target_train=tar_tokenizer.texts_to_sequences(decoder_target_train)
decoder_input_test=tar_tokenizer.texts_to_sequences(decoder_input_test)
decoder_target_test=tar_tokenizer.texts_to_sequences(decoder_target_test)
#현재 준비된 데이터들: (encoder_input_train, encoder_input_test), (decoder_input_train, decoder_input_test), (decoder_target_train, decoder_target_test)

 #5. 빈 샘플 제거
#빈도수가 낮은 단어가 삭제되면서 빈 샘플의 존재 가능성(특히 Summary). 주의할 점은 sostoken과 eostoken은 모든 문장에 등장하기에 논리적으로 빈도수 기반으로 컷되지 않는다. 즉, decoder의 실질적인 empty sample의 길이는 1일 것이다.
drop_train=[index for index, sentence in enumerate(decoder_input_train) if len(sentence)==1]#decoder_input_train의 empty sample 제거 (토큰고려 1과 비교)
drop_test=[index for index, sentence in enumerate(decoder_input_test) if len(sentence)==1]
print('삭제할 훈련 데이터 개수: ', len(drop_train))#1235(디코더에서 삭제할거면 인코더도 삭제해야함. 참고로 인코더는 확인 안하는게 Summary에서 차피 극심할거라)
print('삭제할 테스트 데이터 개수: ', len(drop_test))#337

encoder_input_train=np.delete(encoder_input_train, drop_train, axis=0)
decoder_input_train=np.delete(decoder_input_train, drop_train, axis=0)
decoder_target_train=np.delete(decoder_target_train, drop_train, axis=0)

encoder_input_test=np.delete(encoder_input_test, drop_test, axis=0)
decoder_input_test=np.delete(decoder_input_test, drop_test, axis=0)
decoder_target_test=np.delete(decoder_target_test, drop_test, axis=0)

print('빈 샘플 제거 후 \n훈련 데이터의 개수: ', len(encoder_input_train))
print('훈련 레이블의 개수: ' ,len(decoder_input_train))
print('테스트 데이터의 개수: ', len(encoder_input_test))
print('테스트 레이블의 개수: ', len(decoder_input_test))

 #6. 패딩하기
encoder_input_train=pad_sequences(encoder_input_train, maxlen=text_max_len, padding='post')
encoder_input_test=pad_sequences(encoder_input_test, maxlen=text_max_len, padding='post')
decoder_input_train=pad_sequences(decoder_input_train, maxlen=summary_max_len, padding='post')
decoder_input_test=pad_sequences(decoder_input_test, maxlen=summary_max_len, padding='post')
decoder_target_train=pad_sequences(decoder_target_train, maxlen=summary_max_len, padding='post')
decoder_target_test=pad_sequences(decoder_target_test, maxlen=summary_max_len, padding='post')


    #[3. seq2seq + attention으로 요약 모델 설계 및 훈련]
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

embedding_dim=128
hidden_size=256

#인코더
encoder_inputs=Input(shape=(text_max_len,))
enc_emb=Embedding(src_vocab, embedding_dim)(encoder_inputs)

#LSTM층을 3개 쌓는다
encoder_lstm1=LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1=encoder_lstm1(enc_emb)

encoder_lstm2=LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2=encoder_lstm2(encoder_output1)

encoder_lstm3=LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c=encoder_lstm3(encoder_output2)

#디코더
decoder_inputs=Input(shape=(None,))
dec_emb_layer=Embedding(tar_vocab, embedding_dim)
dec_emb=dec_emb_layer(decoder_inputs)

#디코더의 단일 LSTM층
decoder_lstm=LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
decoder_outputs, _, _=decoder_lstm(dec_emb, initial_state=[state_h, state_c])#인코더의 최종 state를 initial_state로 사용. it's called as Context Vector!

#디코더의 출력층
decoder_softmax_layer=Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs=decoder_softmax_layer(decoder_outputs)#이거 바로 사용하는거랑 저장하고 사용하는거랑 차이가 있으려나..epoch마다 새로 할당하려나 -> 이따 테스트용 재설계시 재활용한다..


#모델 정의
model=Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)#이처럼 decoder_softmax_outs을 출력층으로 사용하는 것이 이상적이나, 바다나우 어텐션을 사용할 것이다.
model.summary()


#어텐션 메커니즘이 결합된 새로운 출력층의 설계 by 바다나우 어텐션
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/20.%20Text%20Summarization%20with%20Attention/attention.py", filename="attention.py")
from attention import AttentionLayer

attn_layer=AttentionLayer(name='attention_layer')#인코더 출력, 디코더 입력 어텐션
attn_out, attn_states=attn_layer([encoder_outputs, decoder_outputs])#추가는 concat만 하니 존나게 간단한데 내부적으론 존나게 복잡하겠지..

#어텐션 결과와 디코더의 hidden states를 연결
decoder_concat_input=Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])#약간 residual connection느낌으로 attention 출력이랑 디코더의 출력이랑 concat

#어텐션 연산이 추가된 디코더의 출력층
decoder_softmax_layer=Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs=decoder_softmax_layer(decoder_concat_input)


#어텐션이 추가된 새로운 모델 정의
model=Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)#내부적으로 attention사용
model.summary()

 #모델 컴파일
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
history=model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train, validation_data=([encoder_input_test, decoder_input_test], decoder_target_test), batch_size=256, callbacks=[es], epochs=50)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()#train, test loss visualization


    #[4. seq2seq +attentin으로 요약 모델 테스트하기]
#테스트 준비
src_index_to_word=src_tokenizer.index_word
tar_word_to_index=tar_tokenizer.word_index
tar_index_to_word=tar_tokenizer.index_word

#테스트를 위한 모델 재설계
#인코더
encoder_model=Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_h])

#디코더
decoder_state_input_h=Input(shape=(hidden_size,))#이전 시점 상태를 저장 buffer
decoder_state_input_c=Input(shape=(hidden_size,))

dec_emb2=dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2=decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])#이전의 상태 사용, 이후의 상태 저장 for 재사용

#어텐션
decoder_hidden_state_input=Input(shape=(text_max_len, hidden_size))#(emb * hidden)
attn_out_inf, attn_states_inf=attn_layer([decoder_hidden_state_input, decoder_output2])#인코더 출력 대신 디코더입력, 디코더 출력 어텐션(디코더 마지막 시점에 입력이 인코더의 출력이 아닌 디코더의 입력이었다.)
decoder_inf_concat=Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

#디코더의 출력층
decoder_outputs=decoder_softmax_layer(decoder_inf_concat)

#최종 디코더 모델 완성
decoder_model=Model([decoder_inputs]+[decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c], [decoder_ouputs2]+[state_h2, state_c2])
#[target_seq]+[e_out, e_h, e_c]를 초기값으로 받고, e_h, e_c를 갱신해나간다.


#테스트를 위해 사용되는 함수. 역시 e_out을 사용하지 않는고만. 분명 seq2seq의 인코더 출력은 사용하지 않거든...state의 context vector만을 사용하지
def decode_sequence(input_seq):
    e_out, e_h, e_c=encoder_model.predict(input_seq)#context vector, hidden_state, cell_state

    target_seq=np.zeros((1,1))#sostoken을 디코더의 입력으로 건네주기 위함.
    target_seq[0,0]=tar_word_to_index['sostoken']
    
    stop_condition=false
    decoded_sentence=''
    while not stop_condition:
        output_tokens, h, c=decoder_model.predict([target_seq]+[e_out, e_h, e_c])#target_seq로 sostoken전달.
        sampled_token_index=np.argmax(output_tokens[0, -1, :])
        sampled_token=tar_index_to_word[sampled_token_index]

        if (sampled_token!='eostoken'):
            decoded_sentence+=' '+sampled_token
        if (sampled_token=='eostoken' or len(decoded_sentence.split())>=(summary_max_len-1)):#끝이거나 오약 최대길이 넘게 예측하면 중단
            stop_condition=True

        target_seq=np.zeros((1,1))
        target_seq[0,0]=sampled_token_index#예측한 단어를 디코더의 target_seq로 사용하게 갱신

        e_h, e_c=h, c#디코더의 상태로 갱신 for 다음 디코더 입력

    return decoded_sentence


#테스트 단계에서 원문과 실제 요약문, 예측 요약문을 비교하기 위해 정수 시퀀스를 텍스트로 만드는 각각의 함수 정의
def seq2text(input_seq):
    sentence=''
    for i in input-seq:
        if (i!=0):#0을 안사용해서 그런건감 아 패딩이구나 오호라 난 천재야
            sentence=sentence+src_index_to_word[i]+ ' '
    return sentence
def seq2summary(input_seq):#사용하는 vocab이 다르니..
    sentence=''
    for i in input_seq:
        if (i!=0 and i!=tar_word_to_index['sostoken'] and i!=tar_word_to_index['eostoken']):
            sentence=sentence+=tar_index_to_word[i]+' '
    return sentence

#실제 테스트
for i in range(500, 1000):
    print('원문: ', seq2text(encoder_input_test[i]))
    print('실제 요약문: ', seq2summary(decoder_input_test[i]))
    print('예측 요약문: ', decode_sequence(encoder_input_test[i].reshape(1,text_max_len)))
    print('\n')
