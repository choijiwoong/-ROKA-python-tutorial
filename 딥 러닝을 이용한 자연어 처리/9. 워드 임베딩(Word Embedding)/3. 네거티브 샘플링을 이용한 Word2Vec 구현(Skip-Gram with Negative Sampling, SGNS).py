""" [네거티브 샘플링(Negative Sampling)]
Word2Vec의 출력층은 softmax를 이용지난 뒤 오차를 구하고 모든 단어에 대한 embedding vector값을 update하는데, 이는 무거운 작업이다.
또한 업데이트하려는 단어와 상관없는 단어를 업데이트하는 것은 비효율적이기에 일부 단어 집합에만 학습을 집중시키는데 이를 Negative Sampling이라고 한다.
context word와 상관없는 랜덤 center word를 가져와 positive인지, negative인지를 binary classification을 사용하여 효율적인 연산을 수행한다.

    [네거티브 샘플링 Skip-Gram(Skip-Gram with Negative Sampling, SGNS)]_center word로 context word예측
네거티브 샘플링은 기존에 앞뒤 단어들로 추론한 것과 달리, 앞뒤단어들이 모두 입력이 되고, 이 두 단어가 실제 윈도우 크기 내에 존재하는 이웃 관계인지에 대한 확률을 예측한다.
실제 이웃관계였던 단어들은 레이블을 1로, 랜덤단어들은 0으로 초기화하고, 그 둘을 위한 Embedding layer(table) for lookup of center word을 만든다.
각 단어는 label에 따른 임베딩 테이블을 table lookup하여 embedding vector로 변환되며, 중심단어와 주변 단어의 내적값을 모델의 예측값으로 하고,
레이블(positive, negative)과의 오차로부터 backpropagation하여 embedding vector를 update한다. 학습후에 하나의 embedding layer을 사용하거나
더하여 사용하거나 concatenate하려 사용하면 된다.

 그니까, Skip-gram데이터셋을 둘다 입력으로 그리고 label1로 바꾸어 SGNS의 데이터셋을 만들고, 여기에 랜덤단어들을 label0으로 추가함.
그다음 label별 embedding layer을 2개 만들어 각각 변환한 뒤에 center embedding vector와 (random)context embedding vector를 내적하여 나온 예측값을 해당 label값과
손실계산을 통해 embedding layer을 update하여 학습시킴."""

import plaidml.keras
#plaidml.keras.install_backend()

print('<1단계 도입>')#1. 20뉴스 그룹 데이터 전처리하기]_데이터에 최소 단어 2개가 있어야, 중심단어&주변관계를 성립시킬 수 있다.
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.preprocessing.text import Tokenizer

dataset=fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents=dataset.data
print('총 샘플 수: ', len(documents))

 #regulization & cleaning
news_df=pd.DataFrame({'document': documents})
news_df['clean_doc']=news_df['document'].str.replace("[^a-zA-Z]", " ")#영어 아닌거 공백으로
news_df['clean_doc']=news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))#길이가 3초과면 하나로 합침
news_df['clean_doc']=news_df['clean_doc'].apply(lambda x:x.lower())#소문자화
print("NUll값 여부: ", news_df.isnull().values.any())
news_df.replace("", float("NaN"), inplace=True)#빈(empty)값의 유무도 확인해야하기에 이를 null로 바꾼뒤, null값의 여부를 다시 확인한다.
print('Null값 여부(빈값 NaN화 후): ', news_df.isnull().values.any())
news_df.dropna(inplace=True)
print("null&empty값 제거 후 총 샘플 수: ", len(news_df))
 #불용어 제거
stop_words=stopwords.words('english')
tokenized_doc=news_df['clean_doc'].apply(lambda x: x.split())#split
tokenized_doc=tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])#filtering
tokenized_doc=tokenized_doc.to_list()#restore splited data to list again
 #단어개수 1이하 제거
drop_train=[index for index, sentence in enumerate(tokenized_doc) if len(sentence)<=1]#길이 1이하 index리스트
tokenized_doc=np.delete(tokenized_doc, drop_train, axis=0)
print('길이 1이하 단어 & 불용어 제거 후, 총 샘플 수: ', len(tokenized_doc))

 #단어 집합 생성, Integer encoding
tokenizer=Tokenizer()
tokenizer.fit_on_texts(tokenized_doc)

word2idx=tokenizer.word_index
idx2word={value: key for key, value in word2idx.items()}#tools
encoded=tokenizer.texts_to_sequences(tokenized_doc)#integer encoding
print("정수인코딩된 상위2개의 샘플:", encoded[:2])

vocab_size=len(word2idx)+1
print("단어 집합의 크기: ", vocab_size,'\n')


print('<2단계 도입>')#2. 네거티브 샘플링을 통한 데이터셋 구성하기
from tensorflow.keras.preprocessing.sequence import skipgrams#negative sampling tool.
#현재 encoded에는 20newsgroup에 대한 전처리, 정수인코딩된 상태의 데이터가 있다.
skip_grams=[skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded[:10]]#상위 10개만.

 #첫번째 뉴스그룹 샘플에 대해 negative sampling dataset 결과확인
pairs, labels=skip_grams[0][0], skip_grams[0][1]#첫번째 그룸 샘플 저장(첫번째 그룹의 pairs, 첫번째 그룹의 labels)
for i in range(5):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
        idx2word[pairs[i][0]], pairs[i][0],#word1, index1(첫번째 그룹의 첫번째 샘플의 pairs 단어버전, 숫자버전)
        idx2word[pairs[i][1]], pairs[i][1],#word2, index2
        labels[i]))#label
print('전체 샘플(skip_grams) 수: ', len(skip_grams))#상위 10개의 샘플만 가져옴.
print('첫번째 뉴스그룹의 pair수: ', len(pairs), ', labels수: ', len(labels))

print('모든 encoded에 대하여 skipgrams 적용중...\n\n')
skip_grams=[skipgrams(sample, vocabulary_size=vocab_size, window_size=10) for sample in encoded]#모든 encoded에 수행.


print('<3단계 도입>')#3. Skip-Gram with negative Sampling(SGNS) 구현하기
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Reshape, Activation, Input
from tensorflow.keras.layers import Dot
from tensorflow.keras.utils import plot_model
from IPython.display import SVG
from tqdm import tqdm

#hyper parameter
embedding_dim=100
#embedding table1 for center word
w_inputs=Input(shape=(1,), dtype='int32')#Shape!
word_embedding=Embedding(vocab_size, embedding_dim)(w_inputs)
#embedding table2 for context word
c_inputs=Input(shape=(1,), dtype='int32')#Shape!
context_embedding=Embedding(vocab_size, embedding_dim)(c_inputs)#vocab_size is same with one-hot vector's shape. (input_dim, output_dim)

#process of prediction
dot_product=Dot(axes=2)([word_embedding, context_embedding])#(functinoal modeling..?)두 벡터의 내적
dot_product=Reshape((1,), input_shape=(1,1))(dot_product)#성형as output_dim=1(for layer?), input_dim=1,1
output=Activation('sigmoid')(dot_product)#[0~1]
#modeling
model=Model(inputs=[w_inputs, c_inputs], outputs=output)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True, rankdir='TB')
#training
for epoch in tqdm(range(1,6)):
    loss=0
    for _, elem in enumerate(skip_grams):#skip_grams return pairs, labels. (index, (pairs, labels))
        first_elem=np.array(list(zip(*elem[0]))[0], dtype='int32')#integer encoded된 pairs의 첫번째 단어
        second_elem=np.array(list(zip(*elem[0]))[1], dtype='int32')#integer encoded된 pairs의 두번째 단어
        labels=np.array(elem[1], dtype='int32')#pairs의 label값
        X=[first_elem, second_elem]#두 word의 integer encoded값을 list로 만들어서 X데이터 구성.
        Y=labels
        loss+=model.train_on_batch(X,Y)#
    print('Epoch: ', epoch, 'Loss: ', loss)


print('<4단계 도입>')#4. 결과 확인하기
import gensim

f=open('vectors.txt', 'w')
f.write('{} {}\n'.format(vocab_size-1, embedding_dim))
vectors=model.get_weights()[0]
for word, i in tokenizer.word_index.items():
    f.write('{} {}\n'.format(word, ' '.join(map(str, list(vectors[i, :])))))
f.close()

w2v=gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
print("'soldiers'와 유사한 단어: ", w2v.most_similar(positive=['soldiers']))
print("'Ulsan'과 유사한 단어: ", w2v.most_similar(positive=['Ulsan']))
print("'coke'와 유사한 단어: ", w2v.most_similar(positive=['coke']))
