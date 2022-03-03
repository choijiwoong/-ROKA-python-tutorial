""" [피드 포워드 신경망(Feed-Forward Neural Network, FFNN)]
오직 입력층에서 출력층 방향으로 연결이 되는 것으로, 피드 포워드 신경망이 아닌 것으로 대표적으로 은닉층의 출력값이 다시 은닉층의 입력으로도 사용되는 RNN이 있다.

    [전결합층(Fully-connected layer, FC, Dense layer)]
어떤 층의 모든 뉴런이 이전 층의 모든 뉴런과 연결돼 있는 층을 의미한다. Dense layer로도 부르기도 하며 Keras에서 Dense()가 이것이다.

    [활성화 함수(Activation Function)]
활성화 함수는 출력이 입력의 상수배가 되는 선형 함수가 아닌, 비선형함수여야 하는데 은닉층을 쌓기 위해서이다. 일반적인 선형함수의 경우
wxwxwxwxX가 되는데, 이는 w^4X와 다름이 없기에 은닉층을 여러번 쌓더라도 효과가 없다. 하지만 활성화 함수가 없는 선형함수층을 사용하는 경우도 있는데,
이를 비선형층(nonlinear layer)등과 함께 인공신경망의 일부로서 추가하여 학습 가능한 가중치를 새로 만들고 싶을 때이다. 이와같은 상황에서 선형함수를 이용한 층을 일반적인 은닉층과 구분하기 위해서
선형층(Linear layer), 투사층(Projection layer)등의 표현을 사용하며, 임베딩 층(embedding layer)도 선형층의 일종이다."""
import numpy as np
import matplotlib.pyplot as plt

#[Step function]
def step(x):
    return np.array(x>0, dtype=np.int)#양수면 True, 아니면 False

x=np.arange(-5.0, +5.0, 0.1)
y=step(x)
plt.title('Step Function')
plt.plot(x,y)
plt.show()

#[Sigmoid function]_과 기울기 소실_인공신명망은 forward propagation을 통해 나온 오차를 loss function으로 계산하고 미분으로 gradient를 구한 뒤, 이를 입력층방향으로
#가중치와 편향을 update하는 back propagation을 수행하는데, 시그모이드는 gradient를 구하는데에서 문제가 발생한다.
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5.0, +5.0, 0.1)
y=sigmoid(x)

plt.plot(x, y)
plt.plot([0,0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')
plt.show()#시그모이드의 모양상(기울기 최고 0.25), 역전파 과정에서 0에 가까운 값이 누적해서 곱해지며 거의 기울기를 전파받을 수 없는 기울기 소실(Vanishing Gradient)문제가 발생한다.

#[Hyperbolic tangent function]
x=np.arange(-5.0, 5.0, 0.1)
y=np.tanh(x)

plt.plot(x,y)
plt.plot([0,0], [1.0,-1.0], ':')
plt.axhline(y=0, color='orange')
plt.title('Tanh Function')
plt.show()#하이퍼볼릭탄젠트는 기울기최고1로 전반적으로 시그모이드보다 큰 값이 나와 기울기 소실이 적은 편이다.

#[ReLU]_수식이 간단하고 가장 인기있다.
def relu(x):
    return np.maximum(0,x)

x=np.arange(-5.0, 5.0, 0.1)
y=relu(x)

plt.plot(x, y)
plt.plot([0,0], [5.0, 0.0], ':')
plt.title('Relu Function')
plt.show()#렐루는 입력값이 음수일시 미분이 바로 0이 되어버리기에 뭐 기울기값이 커지거나 작아지는 방향으로 이동하여 다시 x의 값을 갖게 할 수 없어서 이를 Dying ReLU라고 한다.

#[Leaky ReLU]_dying ReLU를 보완하기 위한 것으로, Leaky(새는정도)를 파라미터로 받아 음수일때의 기울기를 설정할 수 있다.
leaky_degree=0.1
def leaky_relu(x):
    return np.maximum(leaky_degree*x, x)

x=np.arange(-5.0, 5.0, 0.1)
y=leaky_relu(x)

plt.plot(x,y)
plt.plot([0,0], [5.0, 0.0], ':')
plt.title('Leaky ReLU function')
plt.show()#음수일때의 Leaky를 0.1로 설정하여 입력값이 음수라도 기울기가 0이 되지 않으면 ReLU가 죽지않게 dying ReLU문제를 해결하였다.

#[Softmax function]_은닉층에서는 ReLU함수를 사용하는 것이 일반적이며, 출력층에서는 시그모이드함수처럼 소프트맥스가 사용된다.
#시그모이드(로지스틱 회귀)는 Binary Classification에, 소프트맥스는 세가지이상의 exclusive선택지중 하나를 고르는 MultiClass Classification에 사용된다.
x=np.arange(-5.0, 5.0, 0.1)
y=np.exp(x)/np.sum(np.exp(x))

plt.plot(x,y)
plt.title('Softmax Function')
plt.show()
