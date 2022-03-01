""" [머신러닝 이란]
기존에 Data를 모델에 넣으면 해답이 나오던것과 달리, Data와 해답을 모델에 넣어 규칙성을 도출한다.

    [머신러닝 훑어보기]
Validation데이터는 모델의 성능을 평가하는 용도가 아닌, 모델의 성능을 조정하는 용도로, 과적합(overfitting)여부를 판단하거나, 하이퍼파라미터 튜닝을 위한 것이다.
 머신러닝의 많은 문제는 분류(Classification)혹은 회귀(Regression)문제이며, 회귀는 Leneare Regression, Logistic Regression(분류문제에 속함..)
그리고 분류는 Binary Classification, Multi-class Classification, Multi-lable Classification으로 나뉜다.
 이진분류문제는 두개중 하나를 선택, 다중클래스분류는 세개이상중 선택, 회귀문제는 분류문제와 같은 경우에서 정답이 몇개로 정해진게 아니라 어떠한 연속적인 범위로 예측되는 것이다.
회귀문제의 예시로는 시계열 데이터(Time Series Data)를 이용한 주가 예측, 생산성 예측등이 있다.(둘다 범위로 나오니)
 머신러닝은 크게 Sipervised Learning, Unsupervised Learning, Self-Supervised Learning이 있다.(feat. 강화학습)
지도학습은 label(정답)과 같이 학습하는 것이며, 비지도학습은 별도의 레이블없이 학습하는 것이며, 자기지도학습은 레이블이 없는 데이터로부터 스스로 레이블을 만들어
학습하는 방법으로 Word2Vec과 같은 워드 임베딩 알고리즘, BERT등이 있다.
 샘플(Sample)과 특성(Feature)은 머신러닝에서 각각 하나의 행, 각각의 독립변수(열)을 의미한다.
 머신러닝에서 Accuracy계산 중 맞춘 결과과 틀린 결과에 대한 세부적인 내용을 알려주는 것이 혼동 행렬(Confusion Matrix)이다.
예측참&실제참_TP(True Positive_정답), 예측참&실제거짓_FP(False Positive_오답), 예측거짓&실제참_FN(False Negative_오답), 예측거짓&실제거짓_TN(True Negative_정답)으로 표기한다.
이 개념을 이용하여 정밀도(Precisiion), 재현율(Recall), 정확도(Accuracy)를 구할 수 있는데, 정밀도는 TP/TP+FP(True분류한것중 실제True비율),
재현율은 FP/TP+FN(실제 True중에서 모델이 True로 예측한 비율)이기에 정밀도와 재현율을 모두 분자가 공통적으로 TP이다. 정확도는 (TP+TN)/(TP+FN+FP+TN)으로
옳게 예측한 비율을 말한다. 하지만, Accuracy만으로 성능을 판단할 수 없는 이유가 모델이 항상 True를 반환하는데 100개의 메일중 2개만 스팸메일이라면
이 말도안되는 모델은 무려 98%의 정확도를 띄게 된다. 
 과소적합(Underfitting), 과적합(Overfitting)의 의미는 머신러닝에서 학습을 적합(fitting)이라고 부르기 때문이며, 과적합을 막기 위해 Dropout, Early Stopping같은
방법이 존재한다. 검증데이터에서 훈련데이터와 비교했을 떄 오차가 증가했다면 과적합 징후이다.

    [선형 회귀(Linear Regression)]
어떤 변수의 값에 따라 특정 변수가 영향을 받는다면 다른 변수의 값을 변하게 하는 변수를 독립변수, 영향을 받는 변수를 종속변수라고 한다.
이때 선형회귀는 한 개 이상의 독립변수 x와 y의 선형관계를 모델링하는데, 독립변수가 1개라면 특별히 단순 선형 회귀(Simple Linear Regression)라고 한다.
단순 선형 회귀 분석은 y=w(weight)x+b(bias)이며 하나의 직선만 표현이 가능하다.
다중 선형 회귀 분석은 y=w1x1+w2x2+...+wnxn+b로 여러 독립변수로 하나의 종속변수를 예측할 수 있다.
 x와y의 관계를 유추하기 위한 수학적인 식을 가설(Hypothesis)라고 하며, 단순선형회귀의 경우 H(x)=wx+b라고 표현한다. 적절한 w와 b를 찾아내는 것이 우리가 해야하는 목표이다.
이러한 과정에서 실제값과 예측값의 오차식을 값을 최소화하는 경우 비용함수(Cost functino), 손실함수(Loss function), 함수의 값을 최소화하거나 최대화하는 경우 목적함수(Objective function)이라고 한다.
회귀의 경우 손실함수로 평균 제곱 오차(Mean Squared Error, MSE)를 사용한다. 평균편차제곱의 평균으로 손실율을 구한다.
 딥러닝희 학습은 비용함수를 최소화하는 w과 b를 찾는 것이 목적인데, 이 때 사용되는 알고리즘이 Optimizer, 최적화 알고리즘이다.
기본적으로 Gradient Descent를 사용하며, 손실함수의 극소값을 찾는 것이다. cost함수의 도함수가 0이 될때까지 w를 업데이트하는데 식으로 w-a*d cost(w)/d w이다.
이때 a는 학습률 learning rate이며, 변화 폭을 의미한다. 고로 학습률이 지나치게 높으면 w가 극소값을 찾지 못하고 발산해버린다.

    [자동 미분과 선형 회귀 실습]"""
#자동 미분
import tensorflow as tf

w=tf.Variable(2.)

def f(w):
    y=w**2
    z=2*y+5
    return z

with tf.GradientTape() as tape:
    z=f(w)
    
gradients=tape.gradient(z, [w])#z의 도함수에 w를 넣은 미분한 값이 담긴다.
print("gradient result: ", gradients, '\n')

#자동 미분을 이용한 선형 회귀 구현
w=tf.Variable(4.0)
b=tf.Variable(1.0)

@tf.function
def hypothesis(x):#Hypothesis as function
    return w*x+b
x_test=[3.5, 5, 5.5, 6]#가설값 체크
print("hypothesis result: ", hypothesis(x_test).numpy())

@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred-y))#차 제곱의 평균
x=[1,2,3,4,5,6,7,8,9]
y=[11,22,33,44,55,66,77,88,99]

optimizer=tf.optimizers.SGD(0.01)#learning rate with SGD

for i in range(301):
    with tf.GradientTape() as tape:#아래의 과정을 기록하여 gradient계산이 가능하게 한다.
        y_pred=hypothesis(x)
        cost=mse_loss(y_pred,y)

    gradients=tape.gradient(cost, [w,b])#기록한 것 중을 통해 gradient를 계산하는데, cost의 도함수의 인자로 [w,b]를 사용한다.

    optimizer.apply_gradients(zip(gradients, [w,b]))#해당 gradient로 [w,b]인자를 update한다.

    if i%10==0:
        print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}".format(i, w.numpy(), b.numpy(), cost))

x_test=[3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())


#케라스로 구현하는 선형 회귀_Sequential, output_dim
#model=Sequential()
#model.add(kerad.layers.Dense(1, input_dim=1))#output_dim, input_dim 꼴로 이용.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

x=[1,2,3,4,5,6,7,8,9]#공부하는 시간
y=[11,22,33,44,53,68,77,87,100]#성적

model=Sequential()#모델 생성
model.add(Dense(1, input_dim=1, activation='linear'))#model에 Dense(activation은 어떤 함수를 사용할 것인지를 의미. 선형회귀)를 추가
sgd=optimizers.SGD(lr=0.01)#최적화 알고리즘세팅
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])#세팅해준 최적화함수이용, 손실함수는 mse, metrics는 평가할 메트릭 목록이다.
model.fit(x, y, epochs=300)#x와y를 기반으로 300회 훈련한다.
print(model.predict([9.3]))

plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()
