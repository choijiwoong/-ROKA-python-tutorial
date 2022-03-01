#다중 선형 회귀(linear): 독립변수가 여러개하면 H(x)=w1x1+w2x2+...wnxn+b로 표현한다.
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X=np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]])
y=np.array([73,82,72,57,34])

model=Sequential()#모델 생성
model.add(Dense(1, input_dim=3, activation='linear'))#모델에 linear함수 추가(input dimention=3)

sgd=optimizers.SGD(lr=0.0001)#optimizer생성
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])#모델 컴파일
model.fit(X, y, epochs=2000)#훈련

print("학습된 입력에 대한 예측값: ", model.predict(X))
X_test=np.array([[20,99,10], [40,50,20]])
print("테스트 입력에 대한 예측값: ", model.predict(X_test))

#다중 로지스틱 회귀(sigmoid):
X=np.array([[0,0], [0,1], [1,0], [0,2], [1,1], [2,0]])
y=np.array([0,0,0,1,1,1])

model=Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.fit(X,y,epochs=2000)

print(model.predict(X))

#인공 신경망 다이어그램: y=sigmoid(w1x1+w2x2+w3x3+...+wnxn+b)는 인공신경망 다이어그램으로 그림과 같이 표현하며, 로지스틱 회귀를 일종의 인공 신경망의 구조로 해석해도 무방하다.
