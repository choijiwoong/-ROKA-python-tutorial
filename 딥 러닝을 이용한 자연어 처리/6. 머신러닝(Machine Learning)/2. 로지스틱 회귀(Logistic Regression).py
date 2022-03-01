"""Binary Classification을 해결하기 위한 대표적인 알고리즘이 로지스틱 회귀(Logistic Regression)이다.
실제 data와 label을 넣고 그래프로 표시해주면, 특정 시점을 기준으로 값이 0에서 1로 변경되기에 일반적인 직선보다 알파벳S그래프가 나오며
대표적인 S자 형태에서 출력의 범위가 0과 1사이인 함수가 시그모이드함수(Sigmoid function)이다.
 Sigmoid함수는 종종 σ로 표현하며, 1/(1+e^(-wx+b))이다. 이는 sigmoid(wx+b)혹은 σ(wx+b)라고 표현한다."""
#Sigmoid의 시각화
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)

plt.plot(x, y, 'g')#x,y,graph
plt.plot([0,0], [1.0, 0.0], ':')#x=0, y=[1,0]에 :선추가
plt.title('Sigmoid Function')
plt.show()

#가중치 w와 편향 b가 출력값에 어떤 영향을 미치는지
x=np.arange(-5.0, 5.0, 0.1)
y1=sigmoid(0.5*x)#x에 직접 곱해지는 weight를 가정.
y2=sigmoid(x)
y3=sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x,y2,'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0,0],[0.0, 1.0], ':')#x=0, y=[0,1]에 :선추가
plt.title('Sigmoid Function with changing weight')
plt.show()

x=np.arange(-5.0, 5.0, 0.1)
y1=sigmoid(x+0.5)#x에 직접 더해지는 bias를 가정. 이를 통해 binary classification의 기준을 조정할 수 있다.
y2=sigmoid(x+1)
y3=sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0,0], [0,1], ':')
plt.title('Sigmoid Function with changing bias')
plt.show()

"""비용함수(Cost function)
MSE평균제곱오차는 로지스틱 회귀의 비용함수로 사용했을 시 Global Minimum이 아닌 Local Minimum(나 이거 미적분시간에 봤어!)에 빠질 가능성이 매우 높다.
고로 새로운 목적함수(objective function)을 찾아야하는데, log함수로 간단하게 표현이 가능하다.
y가 1일때의 cost function을 -log(H(x)), y가 0일때 cost function을 -log(1-H(x))로 표현이 가능한데, 이를 하나의 식으로 묶으면 다음과 같이 표현할 수 있다.
cost(H(x), y)=-[ylog(H(x))+(1-y)log(1-H(x))] 이는 두개의 식을 -로 묶은 것 외에는 특별한 것이 없므며, 결과적으로 로지스틱 회귀의 목적함수는 아래와 같다.
J(w)_목적함수를 의미=-(1/n)* sigma(i=1~n)*[y^ilog(H(x^i))+(1-y^i)log(1-H(x^i))]로 나타낼 수 있으며 이를 Cross Entropy함수라고 한다.
 결론적으로 로지스틱 회귀는 비용 함수로 크로스 엔트로피함수를 사용하며, 가중치를 찾기위해서는 이 함수의 평균을 취한 함수를 사용한다.
