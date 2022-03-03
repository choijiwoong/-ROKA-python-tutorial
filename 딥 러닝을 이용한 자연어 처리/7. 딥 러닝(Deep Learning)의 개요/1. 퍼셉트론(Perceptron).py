""" [퍼셉트론(Perceptron)]
초기형태의 인공신경망으로 다수의 입력으로 하나의 결과를 내보내는 알고리즘이다. 각 입력값이 가중치와 곱해져 인공뉴런에 보내지고,
그 합이 threshold(임계치_세타)를 넘으면 인공뉴런은 출력신호로 1을 출력한다. 이러한 함수를 Step Function이라고 한다. 임계치를 좌변으로 넘기고 bias로 표현할 수 있다.
 뉴런의 출력값을 변경시키는 함수를 Activation Function이라고 하며, 시그모이드, 소프트맥스 등이 해당한다. 로지스틱 회귀 모델이 인공신경망에서 하나의 인공뉴런이다. 차이는 활성화함수이다.

     [단층 퍼셉트론(Single-Layer Perceptron)]
컴퓨터에서 0과1을 입력해 하나를 출력하는 회로를 게이트(gate)라고한다."""
def AND_gate(x1, x2):
    w1=0.5
    w2=0.5
    b=-0.7

    result=x1*w1+x2*w2+b

    if result<=0:
        return 0
    else:
        return 1
print("AND_gate(+0.5, +0.5, -0.7): (0,0): ", AND_gate(0,0), ", (0,1): ", AND_gate(0,1), ", (1,0): ", AND_gate(1,0), ", (1,1): ", AND_gate(1,1))

def NAND_gate(x1, x2):
    w1=-0.5
    w2=-0.5
    b=0.7

    result=x1*w1+x2*w2+b

    if result<=0:
        return 0
    else:
        return 1
print("NAND_gate(-0.5, -0.5, +0.7): (0,0): ", NAND_gate(0,0), ", (0,1): ", NAND_gate(0,1), ", (1,0): ", NAND_gate(1,0), ", (1,1): ", NAND_gate(1,1))

def OR_gate(x1, x2):
    w1=0.6
    w2=0.6
    b=-0.5

    result=x1*w1+x2*w2+b

    if result<=0:
        return 0
    else:
        return 1
print("OR_gate(+0.6, +0.6, -0.5): (0,0): ", OR_gate(0,0), ", (0,1): ", OR_gate(0,1), ", (1,0): ", OR_gate(1,0), ", (1,1): ", OR_gate(1,1))
"""이 외에도 이들을 충족하는 다양한 가중치와 편향의 값이 있지만, XOR은 구현이 불가능하다. 단층 퍼셉트론은 직선 하나로 두 영역을 나눌 수 있는 문제에 대해서만 구현이 가능하기 때문이다.
고로 직선 두개로 나눌 수 있는 XOR게이트의 경우 다층 퍼셉트론으로 구현한다.

    [다층 퍼셉트론(MultiLayer Perceptron, MLP)]
XOR게이트는 AND, NAND, OR게이트의 조합(퍼셉트론의 층을 더 쌓는다)으로 만들 수 있는데, 이렇게 입출력층 사이의 층을 은닉층(hidden layer)라고 한다.
고로 이러한 방법으로 XOR게이트를 만들어보면 아래와 같다. 은닉층이 2개 이상인 신경망을 심층 신경망(Deep Neural Network, DNN)이라고 한다.
이때 올바른 가중치를 스스로 찾아내개하는게 training, learning이며, loss function과 optimizer을 사용하며, 이들의 대상이 DNN일 경우 Deep Learning이라고 한다."""
def XOR_gate(x1, x2):
    return AND_gate(NAND_gate(x1,x2), OR_gate(x1,x2))
print("XOR_gate(+0.6, +0.6, -0.5): (0,0): ", XOR_gate(0,0), ", (0,1): ", XOR_gate(0,1), ", (1,0): ", XOR_gate(1,0), ", (1,1): ", XOR_gate(1,1))
