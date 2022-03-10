""" Text Classification은 텍스트가 어떤 종류의 범주에 속하는지 구분하는 것으로, Binary Classification, Multi-Class Classification,
Sentiment Analysis, Intent Analysis(명령, 거절, 질문 등)이 있다. RNN계열의 Vanila RNN, LSTM, GRU등을 학습할 예정이다.

    [1. 케라스를 이용한 텍스트 분류 개요(Text Classification using Keras)]
model.add(SimpleRNN(hidden_units, input_shape=(timesteps, input_dim)))에서 hidden_units은 RNN출력의 크기 즉, 은닉 상태의 크기 자체를 의미한다.
timesteps은 시점의 수로 문서 분류에 사용될 경우 입력 시퀀스에 사용되기에 각 문서에서 단어 수에 해당한다.
input_dim은 입력의 크기로 단어별로 입력될 벡터의 크기 즉, 임베딩 벡터의 차원을 의미한다.
 텍스트 분류는 RNN의 many-to-many에 속하며, 모든 timestep에 입력을 받지만 최종 RNN셀만이 은닉상태가 출력층의 활성화함수를 거쳐 분류하는 과정으로 이루어진다.
 분류 클래스의 개수가 N개라면 출력층에 해당하는 Dense layer의 크기를 N으로 한다. 즉, Dense layer은 그전까지의 입력을 결국 분류해야하는
 클래스의 개수만큼 벡터차원을 가지게 하여 각 클래스별 가능성을 나타내기 위한 전처리?에 해당한다."""
    #[2. 스팸 메일 분류하기(Spam Detection)]
 #1. 스팸 메일 데이터에 대한 이해
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/10.%20RNN%20Text%20Classification/dataset/spam.csv", filename="spam.csv")
data=pd.read_csv('spam.csv', encoding='latin1')
print('총 샘플의 수: ', len(data))
print("(test)상위 5개의 샘플 출력: \n", data[:5])

#(전처리)현재 csv에는 NaN의 열 3개가 존재하므로 제거하고, ham과 spam을 label로 사용하기 위해 0과 1로 치환한다.
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1']=data['v1'].replace(['ham', 'spam'], [0,1])
print("(test)전처리 후 상위 5개 샘플 출력:\n", data[:5],'\n\n데이터프레임의 정보:')

#데이터프레임의 정보 표시. non-null정보를 통해 Null값의 샘플이 없다는 것을 확인할 수 있다.
data.info()

