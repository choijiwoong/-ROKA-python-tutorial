""" GloVe(Global Vectors for Word Representation)은 단어 임베딩 방법론으로 카운트 기반의 LSA(Latent Semantic Analysis)와 예측 기반의 Word2Vec의 단점을
보완하여 Word2Vec성능과 유사하다.

    1. 기존 방법론에 대한 비판
LSA는 DTM, TF-IDF행렬처럼 frequency입력으로 차원을 축소(Truncated SVD)하여 잠재된 의미론을 끌어내는 방법론이며, Word2Vec은 실제값과 예측값의 손실로 학습하는 예측 기반의 방법론이다.
LSA는 단어 의미의 Analogy task(유추작업)은 떨어지고, Word2Vec은 Analogy task에서 LSA보다 뛰어나지만, window크기내의 주변 단어만 고려하기에 corpus의 전체적인 통계정보를 반영하지 못한다.

    2. 윈도우 기반 동시 등장 행렬(Window based Co-occurence Matrix)
행과 열을 vocabulary단어들로 구성하고, i단어의 window size내에서 k단어가 등장한 횟수를 [i][k]에 기록한 행렬이다.
쉽게 말해 행,열2개의 단어index에서 해당 단어의 양옆window문장에서 서로 같은게 나온 수를 센 행렬이다.
 이는 Transpose해도 동일한 행렬이 된다는 특징이 있다(i단어의k빈도는 k단어의 i빈도와 같기에)

    3. 동시 등장 확률(Co-occurrence Probability)
P(k|i)는 동시 등장 행렬로부터 특정 단어i의 전체 등장 횟수를 카운트하고, 특정 단어 i가 등장했을때 어떤 단어 k가 등장한 횟수를 카운트하여 계산한 조건부 확률이다.
이는 center word를 i, contect word를 k로, i행k열값/중심단어i행의 합이다.

    4. 손실 함수(Loss function)
GloVe는 임베딩 된 중심 단어와 주변 단어 벡터의 내겆이 전체 코퍼스에서의 동시등장확률이 되도록 만드는 것이 주 아이디어이다.
두 단어의 동시등장확률의 ratio정보의 차를 이용하고, 중심단어와 주변단어는 자유롭게 교환될 수 있어야하기에 실수의 덧셈과 양수의 곱셈에 대하여
준동형(Homomorphism)_F(a+b)=F(a)F(b)를 만족하도록 해야한다.
 손실함수의 결과(조건부 확률)은 스칼라값이어야하기에 a,b를 각각 벡터값으로 사용하지 않고 내적값으로 바꾼다.
GloVe에 준동형 나눗셈버전_F(a-b)=F(a)/F(b)의꼴을 띄게 변형한다. 그 후 이를 만족하는 F(x)의 값을 찾으면 되는데, Ecponential function이다.
이를 log로 바꿔준다음, 완전하게 두 값의 위치를 바꾸어도 식이 성립하게하기 위해 바꾸면 변해버리는 항을 bias로 대체하여 손실함수를 일반화한다.
그리고 나오는 logXmn항의 0이 될 수  있는 가능성, 그리고 동시등장행렬이 DTM처럼 Sparse Matrix일 가능성등을 고려하여 Xik값에 영향을 받는
가중치 함수를 손실함수에 도입한다. 또한 이 함수의 최댓값(1)을 설정하여 'It is'와 같이 빈도수가 높다고 지나친 가중을 받게하지 않는다.
이 가중치 함수는 min(1, (x/xmax)^3/4)이며, 이를 곱하여 최종적인 손실함수를 얻어낼 수 있다
Loss function=Sigma(m,n=1~V) f(Xmn)(Wm^Tw'n+bm+bn'-logXmn)^2"""

    #5. GloVe 훈련시키기
from glove import Corpus, Glove
#글로브 설치가 안되네...ㅠ

corpus=Corpus()
#make co-occurence matrix
corpus.fit(result, window=5)#result=영어와 한국어 Word2Vec 학습하기의 전처리마친 데이터
glove=Glove(no_components=100, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=20, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

print("man과 유사한 단어리스트:", glove.most_similar("man"))
