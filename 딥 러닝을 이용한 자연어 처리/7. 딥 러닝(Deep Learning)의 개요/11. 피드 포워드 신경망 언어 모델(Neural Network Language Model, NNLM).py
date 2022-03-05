"""번역기, 음성 인식과 같은 자연어 생성(Natural Language Generation, NLG)기반의 언어모델, Statistical Language Model(SLM)역시
인공신경망을 이용한 언어모델들로 대체되기 시작했다. 신경망 언어 모델의 시초는 Feed Forward Neural Network Language Model로, NNLM으로 표현한다.
이에 발전된 RNNLM, BiLM등의 언어모델들도 존재한다.

    1. 기존 N-gram 언어모델의 한계
주어진 문맥으로 아직 모르는 단어를 예측하는 것이 Language Modeling인데, n-gram언어모델은 바로 앞 n-1개 단어를 참고하여 예측한다.
다만 Sparsity problem 즉, 충분한 데이터를 관측하지 못한다면 현실에서 존재가능하나 데이터에 없다는 이유로 확률 자체를 0으로 설정할 수 있다.

    2. 단어의 의미적 유사성
Sparsity problem은 단어의 의미적 유사성을 알면 해결이 가능하다. 즉, 똑같진 않지만 두 단어가 유사한 의미임을 학습한다면 본래 확률이 0으로 설정된
희소한 단어와 유사한 단어로 대신 선택하므로써 자연어 생성이 가능하다. 이러한 의미적 유사성을 고려한 모델이 NNLM이며, 이 아이디어는
단어 벡터 간 유사도를 구할 수 있는 벡터를 얻어내는 워드 임베딩(word embedding)의 아이디어이기도 하다.

    3. 피드 포워드 신경망 언어 모델(NNLM)
예문: "what will the fat cat sit on"  목표: 'what will the fat cat'으로 sit예측

우선 단어를 인식할 수 있게 one-hot encoding
what = [1, 0, 0, 0, 0, 0, 0]
will = [0, 1, 0, 0, 0, 0, 0]
the = [0, 0, 1, 0, 0, 0, 0]
fat = [0, 0, 0, 1, 0, 0, 0]
cat = [0, 0, 0, 0, 1, 0, 0]
sit = [0, 0, 0, 0, 0, 1, 0]
on = [0, 0, 0, 0, 0, 0, 1]

NNLM은 n-gram처럼 window(n)크기만큼만을 고려. 입력은 'will, the, fat, cat'의 원-핫 벡터

4개의 원-핫 벡터는 projection layer을 지나는데, 활성화함수 없이 각 입력 단어들이 VxM 가중치 행렬과 곱해진다. 이 결과는
원-핫 벡터와 가중치 W행렬의 곱이 W행렬의 i번째 행에 저장된 것이기에 lookup table로 사용된다.

원-핫 벡터는 M차원의 벡터로 매핑하는데, 이 벡터는 초기에는 랜덤한 값을 가지지만 학습과정에서 값이 계속 변경되는 embedding vector라고 한다.
이 embedding vector들이 concatenate(그냥 하나하나 연결_붙이기)되어 project layer을 지나간다.

projection layer의 결과는 h크기를 가지는 hidden layer을 지나는데, 특별한게 아니라 이전의 projection layer와 달리 평범하게 가중치 곱한 후 편향을 지나 활성화 함수의 입력이 된다는 의미이다.
그럼 원-핫 벡터들과 동일하게 V차원의 벡터를 얻는데, 출력층에서 softmax를 사용하여 벡터의 각 원소를 0과 1사이의 실수값을 갖게 한다.

이는 다음 단어일 확률을 나타내며, 실제값의 손실계산으로 cross-entropy를 사용한다. 단어집합 모든 단어의 선택지 중 정답인 sit을 예측해야하기 때문이다.
이때 projection layer의 weight matrix로 학습된다.

이를 통해 얻는 것은 유사한 목적으로 사용되는 단어들은 유사한 임베딩 벡터값을 얻기에 앞의 단어만 유사하다면 유사한 단어 집합을 만들 수 있다.

단어간 유사도를 구하는 임베딩 벡터의 아이디어는 Word2Vec, FastText, GloVe등으로 발전되었다.

    4. NNLM의 이점과 한계
sparsity problem은 해결하였지만, n-gram처럼 모든 이전 단어가 아닌 정해진 n개의 단어만을 참고한다는 한계가 있다.
이는 Recurent Neural Network을 사용한 Recurrent Neural Network Language Model(RRNLM)으로 극복가능하다"""
