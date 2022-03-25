"""Topic Modeling은 토픽이라는 문서 집합의 추상적인 주제를 발견하기 위한 통계적 모델 중 하나로, 숨겨진 의미 구조를 발견하기 위한 텍스트 마이닝 기법이다.
Latent Semantic Analysis(LSA) 혹은 Latent Semantic Indexing(LSI)는 토픽 모델링 알고리즘인 LDA의 아이디어를 제공한 알고리즘이다.

 1. Singular Value Decomposition(SVD)
A=U S V.T이며, A가 mxn행렬일 경우 다음과같은 3개의 행렬 곱으로 decomposition하는 것이다.
U: mxm 직교행렬, V: nxn 직교행렬, S: mxn직사각 대각행렬. 여시어 orthogonal matrix는 self*self.transpose가 identity matrix가 되는 행렬이며,
diagonal matrix는 주대각선을 제외한 곳의 원소가 0인 행렬이다. S의 원소의 값을 singular value라고 한다.
(용어정리) 전치 행렬: Transposed Matrix, 단위 행렬: identity_matrix(주대각1), 역행렬: Inverse Matrix(순행렬과 곱하여 identity_matrix되게하는 행렬)
           직교 행렬: Orthogonal Matrix(=Inverse Matrix), 대각 행렬: Diagonal Matrix(직사각일 경우에도 원툴 좌측상단기준. 이들이 singular value)

 2. Truncated SVD
위의 일반적인 SVD를 full SVD라고 하며, LSA에서는 일부 벡터가 삭제된 Truncated SVD를 사용한다.
대각 행렬 sigma의 상위값 t개만 남게 되며, 기존의 행렬복구가 불가능하다. 이 t는 찾고자하는 토픽의 수를 의미하는 hyperparametyer이며, 이를 작게 잡아야 노이즈를 제거할 수 있다.
일부 벡터의 삭제는 데이터의 차원을 줄인다라고도 표현하며, 직관적으로 계산 비용이 낮아지고 불필요정보(노이즈)를 제거한다.(특히 영상처리)
 결론적으로, Truncate SVD를 통해 기존의 full SVD에서 드러나지 않는 심층적인 의미 Latent Sementic Information을 얻을 수 있다.

 3. Latent Semantic Analysis
기존의 DTM, TF-IDF행렬은 단어의 의미를 고려하지 않았지만, LSA에서 기본적으로 truncated SVD를 사용하여 잠재의미를 이끌어낸다."""
#1. Full SVD
import numpy as np

A=np.array([[0,0,0,1,0,1,1,0,0],[0,0,0,1,1,0,1,0,0],[0,1,1,0,2,0,0,0,0],[1,0,0,0,0,0,0,1,1]])#문서별 단어등장빈도수(과일이, 길고, 노란, 먹고, 바나나, 사과, 싶은, 저는, 좋아요)
print('size of DTM: ', np.shape(A))#(4,9)총 4개의 문서, 9개의 vocab

U, s, VT=np.linalg.svd(A, full_matrices=True)#Numpy의 linalg.svd는 특이값 분해의 결과로 대각 행렬이 아닌 특이값의 리스트를 반환한다.
print('U: ')
print(U.round(2))
print('U.shape: ', np.shape(U))#(4,4)

print('S: ')
print(s.round(2))#특이값 리스트!
print('S.shape: ', np.shape(s))#(4,)

S=np.zeros((4,9))
S[:4, :4]=np.diag(s)#s의 대각행렬을 0,0~4,4에 저장
print('대각행렬 S:')
print(S.round(2))#특이값 행렬!
print('S.shape: ', np.shape(S))

print('VT: ')
print(VT.round(2))
print('VT.shape: ', np.shape(VT))#(9,9)크기의 직교 행렬(V의 Transpose)

#UxSxVT=A가 되어야한다. 검토
print('A와 UxSxVT가 같은지: ', np.allclose(A, np.dot(np.dot(U,S), VT).round(2)))

#2. Truncated SVD
S=S[:2, :2]#특이값 상위 2개만 저장
print('\n\n대각 행렬: ')
print(S.round(2))#(2,2)

U=U[:, :2]
print('U:')
print(U.round(2))

VT=VT[:2, :]
print('VT:')
print(VT.round(2))

#축소된 U,S,VT행렬로 곱연산을 하면 값이 손실된 상태이기에 기존의 A행렬을 복구할 수 없다.
A_prime=np.dot(np.dot(U,S), VT)
print(A)
print(A_prime.round(2))#하지만 막상 보면 앵간 비슷한 값이 나오지만, 아예 제대로 복구가 되지 않은 구간도 존재한다.
"""truncated U, S, VT의 의미는
우선 U의 경우 문서x토픽(t)크기이다. 잘렸지만 문서의 개수는 유지중이기에 문서의 상태를 t개의 값으로 표현하고 있다.
즉 이들은 잠재 의미를 표현하기 위한 수치화 된 각각의 문서 벡터라고 볼 수 있다.
 VT는 토픽수(t) x 단어개수이다. 이들의 각 열은 잠재 의미를 표현하기 위해 수치화된 각각의 단어 벡터이다.
즉, U와 VT는 각각 잠재 의미를 내포한 문서벡터, 단어벡터로 볼 수 있다. 이를 통해 다른 문서와의 유사도, 다른 단어와의 유사도, 단어로부터 문서의 유사도 등의 계산이 가능하다."""

