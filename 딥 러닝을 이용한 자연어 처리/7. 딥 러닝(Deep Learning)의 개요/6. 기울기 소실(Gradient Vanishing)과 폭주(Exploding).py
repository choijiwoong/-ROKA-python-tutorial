""" Deep Neural Network에서 역전파 과정에서 입력층으로 갈 수록 Gradient가 작아지며 제대로 업데이트가 안되는 것을 Gradient Vanishing이라고 하며,
기울기가 점차 커지며 발산하는 것을 Gradient Exploding이라고 한다. 이는 RNN에서 쉽게 발생한다. 이들을 막기 위해서는 아래와 같은 방법들이 존재한다.

    1.ReLU와 ReLU의 변형들
sigmoid의 경우 절댓값이 크면 그 기울기가 0에 수렴하기에 Gradient Vanishing이 발생하는데, 은닉층의 activation function으로 sigmoid나 hyperbolic tangent대신
ReLU나 Leaky ReLU등을 사용하는 것이다. 즉 은닉층에서는 sigmoid사용을 지양하자..

    2. 그래디언트 클리핑(Gradient Clipping)
Gradient Exploding을 막기 위해 임계값을 넘지 않도록 값을 자른다. 이는 RNN에서 유용하며 아래와 같이 사용한다."""
from tensorflow.keras import optimizers

Adam=optimizers.Adam(lr=0.0001, clipnorm=1.)#기울기가 1.0을 넘으면 자른다.

""" 3. 가중치 초기화(Weight Initialization)
초깃값만 제대로 설정해줘도 gradient문제를 완화할 수 있는데, 방법은 아래와 같다.
1) 세이비어 초기화(Xavier Initialization) also called as 글로럿 초기화(Glorot Initialization)
Uniform Distribution 또는 Normal Distribution으로 초기화 하는 경우로 나뉘며, 이들은 여러 층의 기울기 분산 사이의 균형을 맞춘다.
이들은 Sigmoid, Hyperbolic tangent함수와 같은 S자 activation function에는 좋지만, ReLU와 같이 사용하면 성능이 좋지 않기에 이들을 사용할 경우에 아래의 He초기화를 사용한다.

2) He 초기화(He Initialization)
이들도 세이비어 초기화처럼 정규분포와 균등분포로 나누는데, He 초기화는 다음층의 뉴런의 수를 반영하지 않는다. ReLU계열에 사용하며, ReLU+He 초기화 방법이 보편적이다.

    4. 배치 정규화(Batch Normalization)_ex) 폭을 줄여 local minimum위험을 감소
(이는 미니 배치에 대해 평균, 분산을 구해 정규화를 하고 이들에 스케일조정γ과시프트β를 통한 선형연산을 사용한다.)
이들은 학습시 배치 단위 평균과 분산으로 이동평균,분산을 저장하고, 테스트시 구해둔 평균과 분산으로 정규화를 한다.
Sigmoid, hyperbolic tangent에서도 기울기 소실 문제가 크게 개선되며, 학습속도, 과적합방지의 장점이 있다.
 다만 미니배치크기에 의존적이기에 배치크기가 작으면 잘 작동을 안하며, RNN처럼 time step마다 다른 통계치를 가지는 경우 적용이 어렵다. 
고로 RNN에 적용하기 위해 Feature단위 정규화가 아닌, layer단위 정규화를 하는 층 정규화(Layer Normalization)를 사용할 수 있다."""
