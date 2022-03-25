import numpy as np

 #1. 매개변수 갱신
class GCD:#일반적인 경사하강법
    def __init__(self, lr=0.01):
        self.lr=lr

    def update(self, params, grads):
        for key in params.key():
            param[key]-=self.lr*grads[key]

class Momentum:#관성을 적용하여 별 변화가 없어도 움직이게끔 momentum을 적용한다.
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None

    def update(self, params, grads):
        if self.v is None:
            self.v={}
            for key, val in params.items():
                self.v[key]=np.zeros_like(val)#업데이트 값을 담을 공간 초기화
        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads[ley]#현재의 값에 momentum을 곱한 후, 경사하강을 수행한 값
            params[key]+=self.v[key]

class AdaGrad:#매개변수에 맞춤형 학습률을 적용하며, 기존 기울기를 단일 곱셈 누산 한 후, 이에 루트 역수를 취해 학습률을 조절한다.
    def __init__(self, lr=0.01):
        self.lr=lr
        self.h=None

    def update(self, params, grads):
        if self.h is None:
            self.h={}
            for key, val in params.items():
                self.h[key]=np.zeros_like(val)

        for key in params.keys():
            self.h[key]+=grad[key]*grads[key]#지금까지의 기울기 곱셈누산
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])-1e-7)#일반적인 경사하강에 억제기(기울기 곱 누산의 역수 루트)장착

class RMSProp:#AdaGrad는 기존 기울기를 계속 누산하기에 무한히 학습하면 갱신강도가 낮아진다는 것을 보완하여 서서히 예전 기울기를 잊고 새 기울기를 크게 반영한다.
    pass

class Adam:#AdaGrad+Momentum
    pass

"""2. 가중치의 초기값
0일 경우 모든 가중치가 똑같이 갱신되기에 여러 가중치의 의미가 사라진다.
랜덤일 경우 0과 1에 활성화 값들이 치우치게 된다. Gradient Vanishing
표준편차 0.01랜덤일경우 0.5부근에 활성화값이 집중되어 다수의 뉴런이 거의 같은 값을 출력하기에 실질적으로 뉴런 1개의 역활정도로 표현력을 제한한다.
Xavier 초기값(표준편차: 1/sqrt(n))의 경우 활성화 값들이 올바르게 분포된다. ReLU의 경우 He초기값(sqrt(2/n))을 사용하는데, ReLU는 음수인영역이 없으니 표준편차를 늘려 양수를 더 많이 보게끔 한 것이다.

  3. 배치 정규화: y=rx+b(r은 확대를, b는 이동을 담당) 이 처리를 활성화 함수 앞 혹은 뒤에 삽입하여 데이터분포를 덜 치우치게 scale과 shift를 수행한다.

  4. 가중치 감소
손실함수에 L2노름(가중치의 제곱 노름)을 더하는 방법이 있다. 1/2lambdaW^2를 더하여 가중치의 확대를 억제하는데, lambda는 정규화의 세기를 조절하는 하이퍼파라미터에 해당한다.
(손실함수에 더하기에 lambda가 커질수록 억제된다) L2노름은 각 원소의 제곱합의 루트를 의미하며, 그 외에 L1노름은 절댓값의 합, Linf노름은 월소 절댓값중 큰 것을 의미한다.
"""
 #5. Dropout(뉴런을 임의로 삭제)
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio=dropout_ratio
        self.mask=None

    def forward(self, x, train_flag=True):
        if train_flag:#훈련중에 임의 뉴런 삭제
            self.mask=np.random.rand(*x.shape)>self.dropout_ratio
            return x*self.mask
        else:#사용중엔 뉴런 모두 사용
            return x*(1.0-self.dropout_ratio)

    def backward(self, dout):
        return dout*self.mask#곱셈이기에 미분은 바뀌어 곱한다.

"""6. 하이퍼파라미터 최적화
모든 후보들의 성능을 확인하는 GridSearch의 경우 시간이 오래걸리고, RandomSearch는 정확도가 다소 떨어질 수 있다. 고로 Bayesian Optimization을 사용하는데,
사전 정보를 최적값 탐색에 반영하는 것으로 미지의 목적 함수에 대한 확률적인 추정하는 모델인 Surrogate Model, 추정결과로 다음번 탐색 후보를 추천하는 Acquisition Function이 필요하다.
 Surrogate Model은 하이퍼파라미터 집합과 일반화된 성능의 관계를 모델링 하며, 이를 기반으로 새로운 하이퍼파라미터 집합이 주어졌을때 일반화된 성능을 예측하는 것이다
이 과정에서 다음 후보로 적합한 하이퍼파라미터 집합을 구하는데 사용되는 것이 Acquition Function이며, 유용하다고 판단된 하이퍼 파라미터 집합과 실제 일반화 성능을 확보해준다. 즉 이 관계를 Surrogate Model이 모델링하는 것이다.
