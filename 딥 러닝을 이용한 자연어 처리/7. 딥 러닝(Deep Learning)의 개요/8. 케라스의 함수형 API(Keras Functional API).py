#Sequential API를 사용한 이전의 설계방식은 복잡한 모델을 만드는데 한계가 있기에 복잡한 모델의 경우 Functional API를 사용할 수 있다.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#기존의 Sequential 모델은 층을 단순히 쌓기만 한다.
model=Sequential()
model.add(Dense(3, input_dim=, activation='softmax'))

#Functional API는 각 층을 일종의 함수로 정의하며, 각 함수를 조합하기위한 연산자를 제공한다.
    #1. 전결합 피드 포워드 신경망(Fully-connected FFNN)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs=Input(shape=(10,))#입력 데이터 크기(shape)로 입력층을 정의한다.
hidden1=Dense(64, activation='relu')(inputs)#은닉층1 추가
hidden2=Dense(64, activation='relu')(hidden1)#은닉층2 추가
output=Dense(1, activation='sigmoid')(hidden2)#출력층 추가

model=Model(inputs=inputs, outputs=output)#하나의 모델로 구성
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])#모델 생성 후에는 sequential과 같이 사용.
#model.fit(data, labels)

inputs=Input(shape(10,))#p.s 층의 이름이 같을 때
x=Dense(8, activation='relu')(inputs)
x=Dense(4, activation='relu')(x)
x=Dense(1, activation='linear')(x)
model=Model(inputs, x)#변수명이 모두 같을때도 마지막 output이 x면 그대로 input과 output을 순서대로 입력.

    #2. 선형 회귀(Linear Regression)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimziers
from tensorflow.keras.models import Model

X=[1,2,3,4,5,6,7,8,9]
y=[11,22,33,44,53,66,77,87,95]

inputs=Input(shape=(1,))
output=Dense(1, activation='linear')(inputs)
linear_model=Model(inputs, output)

sgd=optimizers.SGD(lr=0.01)

linear_model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
linear_model.fit(X, y, epochs=300)

    #3. 로지스틱 회귀(Logistic Regression)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs=Input(shape=(3,))
output=Dense(1, activation='sigmoid')(inputs)
logistic_model=Model(inputs, output)

    #4. 다중 입력을 받는 모델(model that accepts multiple inputs)_p.s 다음과 같이 모델을 만들수도 있다. model=Model(inputs=[a1,a2], outputs=[b1,b2,b3])
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

inputA=Input(shape=(64,))#두개의 입력층을 정의
inputB=Input(shape=(128,))

x=Dense(16, activation="relu")(inputA)#첫번째 입력층에서 분기되는 model(ANN)을 정의
x=Dense(8, activation='relu')(x)
x=Model(inputs=inputA, outputs=x)

y=Dense(64, activation='relu')(inputB)#두번째 입력층에서 분기되는 ANN정의
y=Dense(32, activation='relu')(y)
y=Dense(8, activation='relu')(y)
y=Model(inputs=inputB, outputs=y)

result=concatenate([x.output, y.output])#두개의 ANN을 연결 by concatenate of tensorflow.keras.layers

z=Dense(2, activation='relu')(result)#두 연결층이 연결되어 있는 result의 출력을 2개로
z=Dense(1, activation='relu')(z)#1개로
model=Model(inputs=[x.input, y.input], outputs=z)#각각의 입력층을 inputs으로, output은 concatenate된 z로. 두개의 ANN의 각각입력을 한번에 list로 받아 z를 출력

    #5. RNN(Recurrence Neural Network) 은닉층 사용하기 ** 다음에 자세히 나올 예정. LSTM으로 만든다는 것만 알면 될듯 by functional API
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

inputs=Input(shape=(50,1))#하나의 feature가 50개인 time-step을 입력으로 받는 모델.

lstm_layer=LSTM(10)(inputs)#inputs을 받는 LSTM(output_dim=10).
#x=LSTM(10)
#lstm_layer=x(inputs)과 동일.

x=Dense(10, activation='relu')(lstm_layer)#relu만 추가하여 다시 output_dim=10

output=Dense(1, activation='sigmoid')(x)#sigmoid를 통해 output_dim=1

model=Model(inputs=inputs, outputs=output)


    #6. 동일한 표기
result=Dense(128)(input)#이 문장은 아래와 같다.

dense=Dense(128)
result=dense(input)
