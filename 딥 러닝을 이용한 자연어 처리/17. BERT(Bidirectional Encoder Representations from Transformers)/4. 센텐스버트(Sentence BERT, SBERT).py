"""
 1. BERT의 문장 임베딩
여러 방법이 있지만, 주로 [CLS]토큰의 출력 벡터를 문장 벡터로 간주하거나, avg_pooling을 통한 모든 출력 벡터들의 평균, max_pooling을 통한 모든 출력 벡터들의 최대값을 사용한다.
avg_pooling의 경우 문장 전체 의미에 초점을, max_pooling의 경우 문장의 중요한 의미에 초점을 맞춘다. [CLS]는 입력 문장의 총체적 표현으로 간주하기에 이 자체를 입력문장의 벡터로 간주할 수 있다.

 2. SBERT(Sentence BERT)
[문장 쌍 분류 태스크로 파인 튜닝]
학습을 위해 Natural language Inference(문장 분류 쌍 태스크)를 풀게한다. 이는 Eatilment or Contradiction or Neutral의 relation을 판단하는 문제이다.
SBERT는 Sentence들을 문장 임베딩 한 후 다음의 식을 통해 벡터를 얻는다. h=(u; v; |u-v|) 이 세가지의 값을 concatenate하기에 dimention은 3n이 되며
이를 Dense Softmax의 출력층으로 보내 Multi-classification 을 수행하게 한다. (분류클래스k개라면 Dense에서 Weight_matrix(3n x k)크기를 통과시킨다고 봐도 된다. o=softmax(Wh)

[문장 쌍 회귀 태스크로 파인 튜닝]
학습을 위해 Semantic Textual Similarity(문장 쌍 회귀 태스크)를 풀게 한다. 이는 두개의 문장의 의미적 유사성을 구하는 문제로 0~5범위를 가진다.
위의 NLI와 같이 Sentence Embedding으로 얻은 값의 코사인 유사도를 구한다. 그 뒤 레이블 유사도와의 Mean Squared Error(MSE)를 최소화 하는 방식으로 학습한다.
(참고로 코사인 유사도의 범위는 -1~1이기에 레이블 스코어(0~5)를 5으로 나누어 (0~1)로다가 범위를 바꾸어 학습시킬수도 있다.

정리하여 문장 쌍 분류 태스크로 파인튜닝하거나 문장 쌍 회귀 태스크로 파인튜닝할 수 있으며, 아니면 둘 다 학습시킬 수도 있다."""
