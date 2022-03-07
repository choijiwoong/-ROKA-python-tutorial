""" [워드 임베딩(Word Embedding)]
1. 희소 표현(Sparse Representation)
벡터 혹은 matrix가 one-hot vector처럼 대부분 0으로 표현되는 방법을 sparse representation이라고 한다. 이의 단점은 단어의 크기가 벡터의 차원에 비례하여 증가하는 것이며,
이는 공간적 낭비를 일으키며, DTM도 빈도수가 다른 문서에 적다면 마찬가지로 공간적 낭비를 일으키기에 DTM역시 sparse matrix이다. 이러한 희소벡터의 공통적인 문제점은 단어의 의미를 표현하지 못한다는 것이다.

2. 밀집 표현(Dense Representation)
sparse representation의 반의어로 벡터의 차원을 단어 집합의 크기가 아닌 사용자설정 값으로 맞추며, 실수값을 가지게 된다.
이는 sparse representation과 비교하여 벡터의 차원이 조밀해졌다고 하여 Dense vector라고 한다.

3. 워드 임베딩(Word Embedding)
단어를 Dense vector로 표현하는 것을 word embedding이라고 하며, 이 결과를 embedding vector라고 한다.
워드 임베딩 방법론으로 LSA, Word2Vec, FastText, Glove등이 있으며, keras의 Embedding()은 단어를 랜덤한 값을 가지는 Dense vector로 변환한뒤에
ANN의 가중치 학습과 같이 단어 벡터를 학습하는 방법을 사용한다.

    [워드투벡터(Word2Vec)]
one-hot vector의 단점인 유의미한 유사도를 해결하기 위한 대표적인 방법으로 한국-서울+도쿄=일본, 박찬호-야구+축구=호나우드 같은 유의미한 연산을 가능케한다.
학습 방식으로 주변단어(context word)로 중간단어(center word)를 예측하는 CBOW(Continuous Bag of Words), 중간단어(center word)로 주변 단어(context word)를 예측하는 Skip-Gram이 있다. 
1. 희소 표현(Sparse Representation)
sparse representation의 대안으로 단어의 의미를 다차원 공간에 벡터화하는 분산표현(distributed representation)을 사용하며, 이 분산표현으로
단어 간 유사성을 vertorizing하는 것을 embedding이라고 하며 이를 embedding vector라고 한다.

2. 분산 표현(Distributed Representation)
분포 가설(ditributional hypothesis), 즉 '비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다'라는 가정하에 표현하는 방법이다.
유사한 의미의 단어벡터들이 유사한 벡터값을 가지면 vocabulary의 크기보다 벡터의 차원이 상대적으로 저차원으로 줄어든다.
즉, sparse representation은 고차원에 각 차원을 분리시킨다면, distributed representation은 저차원에 단어의 의미를 여러 차원에 분산하여 표현하며,
단어 벡터 간 유의미한 유사도를 계산할 수 있게끔 한다.

3. CBOW(Continuous Bag of Words)
중심 단어를 예측하기 위한 앞,뒤의 범위를 window라고 하며, window의 크기가 n이라면 주변단어의 개수는 2n이 된다.
윈도우 크기 결정 후, 윈도우를 옆으로 움직여 학습 데이터셋을 만드는 방법을 sliding window라고 한다.
 CBOW는 shallow neural network로 Input layer에 context word들의 one-hot vector가 들어가고 은닉층은 활성화함수없이 lookup table을 담당하는 projection layer이다.
투사층의 크기M에 따라서 임베딩 벡터의 차원이 결정되며, Output layer에서 기존의 입력과 같은 방식으로 도출되야하기에 입력층과 projection layer사이의 가중치행렬이 VxM이라면,
projection layer와 output layer사이의 가중치 행렬은 MxV크기를 가진다. 이는 비슷할 뿐 서로 다른 행렬이다.
 projection layer가 lookup table의 역활을 한다는 것은, 원핫 벡터는 사실상 하나(i)만 1이기에 이를 가중치와 곱하면, 실질적으로 projection layer의 i번째 행을 그대로 읽어오는 것과 같다.
고로 lookup전, 후로 곱해지는 가중치를 잘 훈련시키는 것이 우리의 목표이다.(올바른 값을 lookp할 수 있게)
 여러 입력이 투사층에서 만나 이들의 평균 벡터를 구하게 되는데, 이는 우리가 context word로 centor word하나를 찾는데에 사용하는 기법이기 때문이다.
(CBOW가 아닌 Skip-Gram의 경우 centor word로 context word를 추론하기에 projection layer에서 평균을 구하지 않는다.)
projection layer에서 나온 벡터는 softmax를 통해 일종의 score vector로 바뀌며, label의 one-hot vector와의 오차를 중심으로 학습한다.
 학습이 다 된 후, M차원(projection layer의 차원)의 W행렬의 행을 embedding vector로 사용하거나, 전후 weight matrix를 가지고 embedding vector를 사용한다.
    ㄴ이건 뭔 개소린지 아직은 모르겠음!

4. Skip-gram
centor word로 context word를 predict하기에 평균은 필요없고, 전반적으로 Skip-gram보다 CBOW가 성능이 좋다. 근데 당연하겠지 힌트가 많은데

5. NNLM vs Word2Vec
NNLM은 단어 벡터간 유사도를 위해 word embedding의 개념을 도입하였고, 느린 학습 속도와 정확도를 개선한 것이 Word2Vec이다.
다음단어를 예측하는 NNLM과 달리 Word2Vec은 예측 전, 후의 단어를 모두 참고하여 centor word를 예측하며
Word2Vec은 NNLM의 활성화함수가 있는 hidden layer가 없다. 이로서 빠른 속도를 가지며, hierachical softmax, negative sampling과 같은 좋은 기법으로 우위를 가진다.
 둘의 연산량을 비교하면 아래와 같다.
NNLM: (nxm)+(nxmxh)+(hxV)
Word2Vec: (nxm)+(mxlog(V))"""
