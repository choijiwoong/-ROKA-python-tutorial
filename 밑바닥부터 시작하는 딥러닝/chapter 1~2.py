    #[1. 헬로 파이썬]
 #0. numpy
import numpy as np

A=np.array([[1,2], [3,4]])#배열의 정보가 담겨있는 기존 list데이터를 np.array의 인자로.
print(A)#행, 열을 구분해서 가시성있게 출력해준다.
print(A.shape)
print(A.dtype)

 #1. 원소접근
import numpy as np

X=np.array([[51,55], [14,19], [0,4]])
print(X)

X=X.flatten()
print(X)
print(X[np.array([0,2,4])])#인덱스가 0, 2, 4인 원소가져오기
print(X[X>15])#X값이 15가 넘는 항목들만 가져오기

 #2. pyplot의 기능
import numpy as np
import matplotlib.pyplot as plt

x=np.arange(0,6,0.1)
y1=np.sin(x)
y2=np.cos(x)

plt.plot(x,y1, label='sin')#X와Y데이터를 순서대로 넣으면된다.
plt.plot(x,y2, linestyle='--', label='cos')
plt.xlabel('X')
plt.ylabel('y')
plt.title('sin & cos')
plt.legend()#범례 표시하기
plt.show()

    #[2. 퍼셉트론]
 #1. 퍼셉트론의 단순한 구현
#가중치가 각 입력신호가 결과에 주는 영향력을 조절하는 매개변수라면 임계값theta(편향)은 뉴런이 얼마나 쉽게 활성화(1)이 되는가를 조정하는 매개변수이다.
def AND_(x1, x2):
    w1, w2, theta=0.5, 0.5, 0.7
    temp=x1*w1+x2*w2
    if temp<=theta:
        return 0
    elif temp>theta:
        return 1
print(AND_(0,0), AND_(1,0), AND_(0,1), AND_(1,1))
 #2. 가중치와 편향 구현
def AND(x1, x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.7#세타를 좌변으로 넘기면 -가 붙기에 기존 bias,7에 -를 붙여 b를 -0.7로 세팅
    temp=np.sum(w*x)+b#np.array의 element-wise(원소수동일)를 이용한 입력값과 가중치의 곱. 그리고 더해진 편향.
    if temp<=0:#세타를 넘겨 활성화 기준은 0
        return 0
    else:
        return 1
def NAND(x1, x2):
    x=np.array([x1, x2])
    w=np.array([-0.5, -0.5])
    b=0.7
    temp=np.sum(x*w+b)
    if temp<=0:
        return 0
    else:
        return 1
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.2
    temp=np.sum(w*x)+b
    if temp<=0:
        return 0
    else:
        return 1

 #3. 퍼셉트론의 한계 및 극복: 하나의 perceptron은 결국 linear graph이기에 단층 퍼셉트론으로는 비선형 영역을 분리할 수 없다.
def XOR(x1, x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)#NAND와 OR의 출력을 AND의 입력으로
    return y
print(XOR(0,0),XOR(0,1),XOR(0,1),XOR(1,1))
#그런데 NAND게이트의 조합으로 게이트를 넘어 반가산기, 전가산기, 산술논리장치(ALU), CPU즉 사실상 컴퓨터를 만들 수 있다.
