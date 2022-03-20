 #1. 데이터 로드하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow_datasets as tfds
import tensorflow as tf

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')#Q와 A로 이루어진 데이터

print('null값 유무: ', train_data.isnull().sum())#False

questions=[]
for sentence in train_data['Q']:
    sentence=re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence=sentence.strip()
    questions.append(sentence)
answers=[]
for sentence in train_data['A']:
    sentence=re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence=sentence.strip()
    answers.append(sentence)
print('상위5Q: ', questions[:5])
print('상위5A: ', answers[:5])

 #2. 단어 집합 생성
tokenizer=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions+answers, target_vocab_size=2**13)
START_TOKEN, END_TOKEN=[tokenizer.vocab_size], [tokenizer.vocab_size+1]
VOCAB_SIZE=tokenizer.vocab_size+2
print('시작 토큰 번호: ', START_TOKEN)
print('종료 토큰 번호: ', END_TOKEN)
print("단어 집합의 크기: ", VOCAB_SIZE)

 #3. 정수 인코딩과 패딩
print('임의의 질문 샘플을 정수 인코딩: ', tokenizer.encode(questions[20]))#.encode()로 integer encoding이가능하다.

sample_string=questions[20]#encode(), decode() test
tokenized_string=tokenizer.encode(sample_string)
print('정수 인코딩 후의 문장: ', tokenized_string)
original_string=tokenizer.decode(tokenized_string)
print('기존 문장: ', original_string)

#각 정수가 어떻게 mapping되었는지 확인
for ts in tokenized_string:
    print(ts,'------->', tokenizer.decode([ts]))

#토큰화, 정수인코딩, 시작&종료토큰 추가, 패딩
MAX_LENGTH=40
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs=[], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1=START_TOKEN+tokenizer.encode(sentence1)+END_TOKEN
        sentence2=START_TOKEN+tokenizer.encode(sentence2)+END_TOKEN

        tokenized_inputs.append(sentence1)
        tokenized_outputs.append(sentence2)

    tokenized_inputs=tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs=tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

    return tokenized_inputs, tokenized_outputs
questions, answers=tokenize_and_filter(questions, answers)
print('questions.shape: ', questions.shape)
print('answers.shape: ', answers.shape)

print('임의 데이터 출력: ', questions[0], '\n', answers[0])

 #4. 인코더와 디코더의 입력, 그리고 레이블 만들기
BATCH_SIZE=64
BUFFER_SIZE=20000

dataset=tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]#디코더의 입력. 마지막 패딩토큰 제거.
    },
    {
        'outputs': answers[:, 1:]#시작토큰 제거.
    },
))
dataset=dataset.cache()
dataset=dataset.shuffle(BUFFER_SIZE)
dataset=dataset.batch(BATCH_SIZE)
dataset=dataset.prefetch(tf.data.experimental.AUTOTUNE)
print('inputs: ', answers[0])
print('dec_inputs: ', answers[:1][:, :-1])#마지막 패딩 토큰이 줄어 길이-1
print('outputs: ', answers[:1][:,1:])

 #5. 트랜스포머 만들기
tf.keras.backend.clear_session()
import transformer as tr

D_MODEL=256#Hyperparameters
NUM_LAYERS=2
NUM_HEADS=8
DFF=512
DROPOUT=0.1

model=tr.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


learning_rate=tr.CustomSchedule(D_MODEL)

optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
    y_true=tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))#label.shape=(batch
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
model.compile(optimizer=optimizer, loss=tr.loss_function, metrics=[accuracy])


EPOCHS=50
model.fit(dataset, epochs=1)#빠른 테스트ㅋㅋ

 #6. 챗봇 평가하기
def preprocess_sentence(sentence):
    sentence=re.sub(r'([?.!,])', r' \1 ', sentence)
    sentence=sentence.strip()
    return sentence

def evaluate(sentence):
    sentence=preprocess_sentence(sentence)

    sentence=tf.expand_dims(START_TOKEN+tokenizer.encode(sentence)+END_TOKEN, axis=0)#토큰양옆 추가, 토큰화
    output=tf.expand_dims(START_TOKEN,0)

    for i in range(MAX_LENGTH):
        predictions=model(inputs=[sentence, output], training=False)

        predictions=predictions[:, -1:, :]#현재 시점의 예측단어.
        predicted_id=tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):#종료토큰이면 예측 중단.
            break

        output=tf.concat([output, predicted_id], axis=-1)#현재 예측단어를 output에 연결

    return tf.squeeze(output, axis=0)

def predict(sentence):
    prediction=evaluate(sentence)

    predicted_sentence=tokenizer.decode([i for i in prediction if i<tokenizer.vocab_size])

    print('Input: ', sentence)
    print('Output: ', predicted_sentence)

    return predicted_sentence

output=predict('씨발련아')
output=predict('집가고싶다.')
output=predict('카페에 가면 기부니가 좋고 돈을 깨져용')
