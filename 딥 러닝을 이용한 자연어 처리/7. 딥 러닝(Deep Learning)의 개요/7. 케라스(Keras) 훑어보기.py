    #[1. 전처리(Preprocessing)]
#1-1. tokenize
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()
train_text="The earth is and awesome place live"

tokenizer.fit_on_texts([train_text])#리스트의 형태로 반환

sub_text="The earth is an great place live"
sequences=tokenizer.texts_to_sequences([sub_text])[0]#숫자 시퀀스로 변환. 아마 [1]가 index일거임

print("정수 인코딩: ", sequences)#불용어(길이2이하)알아서 제거되서 5개인듯
print("단어 집합: ", tokenizer.word_index)#word_index로 vocabulary 확인

#1-2. padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
#padding이 'pre'면 앞에 0을, 'post'면 뒤에 0을 padding한다.
print("\npadded [[1,2,3], [3,4,5,6], [7,8]]: ", pad_sequences([[1,2,3], [3,4,5,6], [7,8]], maxlen=3, padding='pre'), '\n')#인자로 건내진 리스트 처럼 길이가 다를 경우 padding을 통해 길이를 맞춰준다.

    #[2. 워드 임베딩(Word Embedding)]_텍스트 내 단어들을 dense vector로 만드는 것으로, one-hot vector보다 저차원이며, 값이 실수이며 훈련데이터로부터 학습하여 표현한다.
#word embedding을 통해 나온 dense vector는 embedding vector라고도 부르며, 주로 256, 512, 1024등의 차원을 가진다.
#Embedding()을 통해 integer encoding된 단어를 Dense vector로 만들 수 있으며 이는 인공신경망 용어로 embedding layer을 만드는 역활이다.
#2D 정수 텐서를 입력받아 3D Tensor를 리턴한다. model.add(Embeddind(vocab_size, output_dim, input_length))

    #[3. 모델링(Modeling)]_케라스에서 입력층,은닉층,출력층같은 층을 구성하기 위해 Sequential()을 사용한다.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))#전결합층이기에 input_dim명시없이 이전 layer의 output_dim으로 추론가능
print("Dense model: ", model.summary())

    #[4. 컴파일(Compile)과 훈련(Training)]_아래는 RNN을 이용하여 Binary Classification을 하는 전형적인 코드이다.
from tensorflow.keras.layers import SimipleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

vocab_size=10000
embedding_dim=32
hidden_units=32

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])#metrics함수는 훈련을 모니터링하기 위한 지표를 선택하는 것이다.

model.fit(X_train, y_train, epochs=10, batch_size=32)#32데이터마다 update하며 전체를 10회 학습한다. (모델이 데이터에 fit해가는 과정)
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data(X_val, y_val))#verbose는 학습중 출력문구를 설정하며, 0~2가 있다.

    #[5. 평가(Evaluation)와 예측(Prediction)]
model.evaluate(X_test, y_test, batch_size=32)#모델에 대한 정확도 평가
model.predict(X_input, batch_size=32)#임의의 입력에 대한 모델의 출력값 확인(실제 사용)

    #[6. 모델의 저장(Save)과 로드(Load)]
model.save("model_name.h5")#인공신경망 모델을 hdf5파일에 저장

from tensorflow.keras.models import load_model
model=load_model("model_name.h5")#저장해둔 모델을 불러온다.
