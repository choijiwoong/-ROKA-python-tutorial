#   [순전파(Forward Propagation)]_입력->입력층->은닉층->출력층->예측값의 과정
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(2, input_dim=3, activation='softmax'))#임의로 softmax(이진분류모델)

print(model.summary())#모든 parameter개수가 8개로 나옴. 3x2를 연결하는 직선의 weight 6개와 bias2개이다.
#h1=x1w1+x2w2+x3w3+b1,  h2=x1w4+x2w5+x3w6+b2,   [y1, y2]=softmax([h1,h2])

    #[행렬곱으로 병렬 연산 이해하기]
#행렬곱을 통한 병렬 연산 시에 유의해야하는 점은, 인공신경망이 matrix를 통해 4개의 샘플을 동시에 처리하더라도 학습 가능한 매개변수의 수는 여전히 8개라는 것이다.
#이렇게 다수의 샘플을 동시에 처리하여 parameter을 조작하는 것을 batch operation이라고 부른다.

    #[행렬곱으로 다층 퍼셉트론의 순전파 이해하기]
model=Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))#4*8
model.add(Dense(8, activation='relu'))#8*8
model.add(Dense(3, activation='softmax'))#8*3
"""입력층: 4입력 8출력, 은닉층1: 8입력 8출력, 은닉층2: 8입력 3출력, 출력층: 3입력 3출력. 이때의 가중치와 편향의 크기를 층마다 추정해보면
입력층->은닉층1: X(1x4)xW(4x8)+B(1x8)=Y(1x8).
은닉층1->은닉층2: X(1x8)xW(8x8)+B(1x8)=Y(1x8)
은닉층2->은닉층3: X(1x8)xW(8x3)+B(1x3)=Y(1x3)
이들은 모두 6.머신러닝에서 'Input Matrix와 Output Matrix를 이용한 Weight와 Bias의 shape추정'의 규칙을 이용한 것이다.
