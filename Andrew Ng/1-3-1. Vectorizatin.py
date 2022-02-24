"""
로지스틱 회귀: 이진분류문제를 풀기 위한 대표적인 알고리즘. 아래의 시그모이드함수같은 것을 이용하여 특정 구간내이 값을 갖게 한다. 이진로시스틱(true, false)등이 있다.
시그모이드 함수: 여러 x값을 [0,1] 로 변경가능한 e^x/(1+e^x) (x=0일때 y=0.5). 활성화 함수의 일종
소프트맥스 회귀: 입력받은 값을 출력으로 [0,1]로 정규화하며, 총합은 1이되는 e^x/(sigma(e^x)) 다차원vector입력을 각각의 확률값으로 표현하는데 사용.
엔트로피: 불확실성의 척도로 높으면 예측이 어렵다.
크로스엔트로피: 모델링을 총해 구한 분포(p)를 통해 q(값)를 예측하는 -sigma(q logp(x)) 예측이 맞으면 0으로 수렴하는 손실함수의 일종.
경사하강법: x=x-a dx_기존의 값에 미분계수&learning rate를 빼어 x를 옮긴다. -는 x가 극값보다 좌측이든 우측이든 일관적인 이동을 하게 해줌.
활성화 함수: 시그모이드, 약한 렐루, 하이퍼탄젠트, 멕스아웃, 렐루, 이엘루 등이 있다. 입력에 대한 출력값을 비선형으로 만든다.
단순선형회귀분석: y=wx+b, 다중선형회귀분석=w1x2+w2x2+...+b

네크워크 말단의 값을 softmax로 각 클래스의 확률값으로 변환하며, 실제값과의 crossentropy차이를 이용하여 backpropagation을 수행한다.
"""
import numpy as np

a=np.array([1,2,3,4])
print(a)

import time
a=np.random.rand(1000000)
b=np.random.rand(1000000)

tic=time.time()
c=np.dot(a,b)
toc=time.time()

print(c)
print('Vectorized version: ', str(1000*(toc-tic)))


c=0
tic=time.time()
for i in range(1000000):
    c+=a[i]*b[i]
toc=time.time()
print(c)
print("For loop: ", str(1000*(toc-tic)))

#CPU와 GPU에서 SIMD(Single Instruction Multiple Data)명령어(dot같은거)는 병렬화의 장점을 이용하여 벡터화 연산을 보다 빠르게 가능하게 한다.


#np.exp(), np.log(), np.abs(), np.maximum(v,0), v**2, 1/v Numpy내장 함수를 사용하여 병렬화의 장점을 최대한 사룔하자.

