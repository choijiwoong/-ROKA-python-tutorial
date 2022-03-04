    #[손실 함수(Loss function)]_
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. MSE(Mean Squared Error, MSE)_연속형 변수를 예측할 때 사용.
model.compile(optimizer='adam', loss='mse', metrics=['mse'])#loss를 mse로 setting
#compile의 loss는 tf.keras.losses.Loss인스턴스를 호출하므로 아래와 같이 표현할 수도 있다.
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])

#2. Binary Cross-Entropy_출력층에서 시그모이드 함수를 사용하는 Binary Classification의 경우 사용.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['acc'])#위와 동일

#3. Categorical Cross-Entropy_출력층에서 소프트맥스 함수를 사용하는 Multi-Class Classification의 경우 사용.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['acc'])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])#one-hot encoding이 아닌 그냥 integer encoding을 사용하고싶은 경우
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['acc'])


    #[배치 크기(Batch Size)에 따른 경사 하강법]_배치: 가중치 등의 매개변수 조정을 위해 사용되는 데이터의 양(전체 or 일부 등등..)
#1. 배치 경사 하강법(Batch Gradient Descent)_loss계산 시 전체데이터를 한번 고려하기에 epoch가 1번인 것이다. 고로 매개변수 업데이트에 오래걸리며, 메모리를 크게 요구한다.
model.fit(X_train, y_train, batch_size=len(X_train))#batch_size를 X_train의 길이로

#2. 배치 크기가 1인 확률적 경사 하강법(Stochastic Gradient Descent, SGD)_랜덤 데이터에 대해 매개변수 값을 조정하는 것으로, Batch Gradient Descent보다 변경폭이 불안정하고,
#정확도가 낮을 수도 있지만, 하나의 데이터만 저장하기에 자원이 적게 사용된다.
model.fit(X_train, y_train, batch_size=1)

#3. 미니 배치 경사 하강법(Mini-Batch Gradient Descent)_배치 크기를 지정. Stochastic와 Batch Gradient Descent의 타협점.
model.fit(X_train, y_train, batch_size=128)


    #[옵티마이저(Optimizer)]
#1. 모멘텀(Momemtum)_Local minimum에 도착하더라도 값을 조절하여 더 낮은 로컬 미니멈 혹은 글로벌 미니멈으로 가게 함.
momentum=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)

#2. 아다그라드(Adagrad)_매개변수별 다른 학습률을 적용시키는데, 변화가 많은 매개변수의 학습률을 작게, 변화가 적은 매개변수를 크게 설정한다.
tf.keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)

#3. 알엠에스프롭(RMSprop)_Adagrad의 단점인 지나치게 낮은 학습률을 다른 수식으로 대체하여 개선한 것이다.
tf.keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-06)

#4. 아담(Adam)_RMSprop+Momentum
adam=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#사용은 model.compile에서 optimizer에 설정하는데, 단순히 문자열로 'adam', 'sgd', 'rmsprop'으로도 설정이 가능하다
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
#or
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


"""[에포크와 배치 크기와 이터레이션(Epochs and Batch size and Iteration)]
문제를 풀고, 답지를 보며 지식을 업데이트한다는 비유...좋구만.. 몇개의 문제를 기준으로 답지를 볼지를 결정할 수 있는데, 이를 epoch라고 한다.
Epoch는 전체 데이터에 대하여 순전파와 역전파가 완료된 상태로, epoch가 50이란 말은 전체 데이터 단위로 50번 학습했다는 것이다.
Batch_size는 몇 개의 데이터 단위로 매개변수를 update했는지로, Batch 수와 혼용하면 안된다. 2000개의 데이터의 배치크기를 200으로 하면 배치의 수는 10이다. 이때 배치의 수를 Iteration이라고 한다.
Iteration은 한 번의 Epoch를 끝내기 위해 필요한 배치의 수이다. Step으로도 부른다."""
