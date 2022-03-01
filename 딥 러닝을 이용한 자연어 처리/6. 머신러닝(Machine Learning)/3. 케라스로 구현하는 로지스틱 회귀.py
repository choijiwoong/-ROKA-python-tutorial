#독립변수x, 숫자가 10이상이면 1, 미만이면 0을 부여한 레이블 y.
#1개의 실수x로부터 1개의 실수y를 예측하는 매핑관계이므로 Dense의 output_dim=1, input_dim=1로 사용하며, 시그모이드를 사용하기 위해 activation='sigmoid'를 사용하낟.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x=np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1])#x=10부터 1로.

model=Sequential()#make model
model.add(Dense(1, input_dim=1, activation='sigmoid'))#add sigmoid with dimention

sgd=optimizers.SGD(lr=0.01)#make optimizer
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])#compile model with SGD, BinaryCrossEntropy, metrics

model.fit(x, y, epochs=200)#train

plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()

print('predict of [1,2,3,4,4.5]: ', model.predict([1,2,3,4,4.5]))
print('predict of [11,21,31,41,500]: ', model.predict([11,21,31,41,500]))
