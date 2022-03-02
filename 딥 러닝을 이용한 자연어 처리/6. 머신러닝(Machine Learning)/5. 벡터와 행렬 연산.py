#텐서는 파이썬에서 3차원 이상의 배열로 표현한다.
import numpy as np

#0차원 텐서(스칼라)_No Axis!
d=np.array(5)
print('0차원 텐서\n텐서의 차원: ', d.ndim)#dimention. 이 값을 axis(축)의 개수라고 부르기도 한다.
print('텐서의 크기(shape): ', d.shape, '\n')

#1차원 텐서(벡터)
d=np.array([1,2,3,4])#_4차원 벡터이자 1D Tensor. 벡터에서의 차원은 하나의 축에 놓인 원소의 개수를 의미하고, 텐서에서의 차원은 축의 개수를 의미한다.
print('1차원 텐서\n텐서의 차원: ', d.ndim)
print('텐서의 크기(shape): ', d.shape, '\n')

#2차원 텐서(행렬)
d=np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])#2차원 텐서. 행과 열이 존재하는 벡터의 배열(행렬_matrix). 2D Tensor
print('2차원 텐서\n텐서의 차원: ', d.ndim)
print('텐서의 크기(shape): ', d.shape, '\n')

#3차원 텐서(다차원 배열)_3D Tensor, 본격적인 텐서(데이터 사이언스 분야 한정으로도 3차원 이상의 배열을 텐서로 부른다)
d=np.array([
        [[1,2,3,4,5], [6,7,8,9,10], [10,11,12,13,14]],
        [[15,16,17,17,19], [19,20,21,22,23], [23,24,25,26,27]]
         ])
print('3차원 텐서\n텐서의 차원: ', d.ndim)
print('텐서의 크기(shape): ', d.shape, end='\n\n\n')
"""[텐서(Tensor)]
3D Tensor는 sequence data를 표현할 때 자주 사용되어 자연어처리에서 단어의 시퀀스를 다룰 때 자주 사용된다. 3D텐서로 samples, timesteps, word_dim이 되는데,
그 외에도 batch_size(샘플의 개수), timesteps(시퀀스의 길이), word_dim(단어를 표현하는 벡터의 차원))으로도 볼 수 있다.
 자연어처리에 사용되는 텐서의 예시로 문서1: I like NLP, 문서2: I like DL, 문서3: DL is AI를 원-핫인코딩하면
I: [100000], like: [010000], NLP: [001000], DL: [000100], is: [000010], AI: [000001]로 표현되는데, 훈련 데이터에 사용되는 단어들을 모두 원-핫 벡터로 바꾸어
한꺼번에 인공신경망의 입력으로 사용한다면
[[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]],
[[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]]
꼴이 되며, 이처럼 훈련 데이터를 다수 묶어 입력으로 사용하는 것을 딥러닝에서 배치(Batch)라고 한다. 위 예시는 (3,3,6)크기를 가지는 3D 텐서이다.
 그 이상의 텐서, 3D Tensor부터는 Cube라고 부르기에, 4D Tensor는 Vector of Cube, 5D Tensor는 Matrix of Cubes라고 부른다.
(정리하면 텐서의 차원은 축의 개수, 3D Tensor부터는 Cube라고 부른다.)

    [케라스에서의 텐서]
케라스에서는 신경망 층에 입력의 크기(Shape)를 줄 때 input_shape인자를 사용하는데, 이는 배치 크기를 제외하고 차원을 지정한다.
즉, input_shape(6,5)라는 인자값을 사용하고 배치크기를 32로 지정하면, 텐서의 크기는 (?,6,5)가 되는 것이다. (배치크기가 지정되기전까진 모르기에)
고로 배치크기까지 같이 지정해주고 싶다면 batch_input_shape=(8,2,10)과 같이 인자를 주면 이 텐서는 (8,2,10)을 의미한다.
 그 외에도 입력의 속성 수를 의미하는 input_dim, 시퀀스 데이터의 길이를 의미하는 input_length등의 인자도 사용하는데, input_shape의 두개의 인자는 (input_length, input_dim)이다.
(정리하면 배치크기지정하려면 input_shape대신 batch_input_shape를 사용하자.)
"""
    #[벡터와 행렬의 연산]
import numpy as np

#같은 크기의 두개의 벡터나 행렬은 element-wisw 덧셈, 뺄셈이 가능하다.
A=np.array([8,4,5])
B=np.array([1,2,3])
print('두 벡터의 합(element-wise): ', A+B)#1D Tensor
print('두 벡터의 차(element-wise): ', A-B, '\n')

A=np.array([[10,20,30,40], [50,60,70,80]])
B=np.array([[5,6,7,8], [1,2,3,4]])
print('두 행렬의 합(element-wise): ')#2D Tensor
print(A+B)
print('두 행렬의 차(element-wise): ')
print(A-B, '\n')

    #벡터의 내적(dot product, inner product)과 행렬의 곱셈
#두 벡터의 차원이 같아야 하며, 두 벡터 중 앞의 벡터가 행벡더(가로 방향 벡터)이고 뒤 벡터가 열벡터(세로 방향 벡터)여야 한다. 고로 내적의 결과는 항상 스칼라가 된다.
A=np.array([1,2,3])
B=np.array([4,5,6])
print('두 벡터의 내적: ', np.dot(A,B), '\n')

#행렬의 곱셈은 왼쪽 행렬의 행벡터와 오른쪽 행렬의 열벡터의 내적이 결과 행렬의 원소가 된다. 행렬 곱셈의 주요한 조건 두가지는 아래와 같다.
#1. 두 행렬의 곱이 성립되기 위해서는 행렬A의 열의 개수와 행렬 B의 행의 개수가 같아야 한다.
#2. 두 행렬의 곱의 결과로 나온 행렬 AB는 A의 행의 개수와 B의 열의 개수를 가진다.
A=np.array([[1,3], [2,4]])
B=np.array([[5,7], [6,8]])
print('두 행렬의 행렬곱: ')
print(np.matmul(A,B))

#행렬의 조건을 익히기 위해 조금 더 연습해볼까..
A=np.array([[1,2,3,4,5], [6,7,8,9,10]])
B=np.array([[1,3,5,7,9], [2,4,6,8,0], [3,5,7,9,11], [4,6,8,10,12], [6,8,10,12,14]])
print('두 행렬의 곱은 아마 shape(2,5): ', np.matmul(A,B).shape)

A=np.array([[1,1,1,1], [2,2,2,2], [3,3,3,3]])
B=np.array([[1,1],[2,2],[3,3],[4,4]])
print('두 행렬의 곱은 아마 shape(3,2): ', np.matmul(A,B).shape)

A=np.array([[1,2,3,4],[1,2,3],[4,5,6,7]])
B=np.array([[1,],[2,],[3,],[4,]])
try:
    print('두 행렬의 곱셈은 안될거같은데?: ', np.matmul(A,B))
except ValueError as er:
    print(er, '\n\n')


""" [다중 선형 회귀 행렬 연산으로 이해하기]  
독립변수가 2개 이상일 때, 1개의 종속변수를 예측하는 다중 선형 회귀나 다중 로지스틱 회귀중 다중 선형 회귀의 경우
y=w1x1+w2x2+...+wnxn+b가 된다. 이때 입력 벡터[x1,...,xn]과 가중치 벡터[w1,...wn]의 내적으로 표현이 가능하며, 반대로 가중치 벡터와 입력벡터의 내적으로도 표현이 가능하다.
 이처럼 샘플의 개수가 많을 때 행렬의 곱셈으로 표현한다. 입력행렬X과 가중치벡터W의 곱+편향벡터B로 전체 가설 수식 H(X)표현이 가능하다.
수학적 관례로 H(X)=WX+B로 표현하기에 가중치*입력 Matrix순으로 matmul하면 가독성이 늘어난다.

    [샘플(Sample)과 특성(Feature)]
머신러닝에서 셀 수 있는 단위로 데이터를 구분할 때, 각각을 샘플로 부르며, 종속변수 y를 예측하기 위한 각각의 독립 변수 x를 특성이라고 부른다.

    [가중치와 편향 행렬의 크기 결정]
행렬곱셈의 조건을 만족해야하기에 입력과 출력의 행렬 크기로부터 가중치 행렬W와 편향 행렬B의 크기를 찾아낼 수 있다.
우선 B와 Y는 같고, matmul의 조건을 따라 X과 W의 맞닿은 부분은 같고, W에서 X와의 공통이 아닌 부분은 Y의 크기와 같아야하기에 추론할 수 있다."""
