"""
    [언어 모델(Language Model)]
언어 모델이란 단어 시퀀스(문장)에 확률(적절한지랄까..)을 할당하는 모델을 말하며, 통계에 기반한 전통적인 언어 모델(Statistical Language Model, SLM)을 알아볼 것이다.
전통언어모델 SLM은 많은 한계가 있지만, 전체적인 개념을 공부하기에 적합하다.

 언어모델(Language Model, LM)은 언어라는 현상을 모델링하고자 단어 시퀀스에 확률을 할당(assign)하는 모델이다. 크게는 통계를 이용하거나 인공 신경망(GPT, BERT))을 이용하여 만들 수 있다.
언어모델을 다른말로 가장 자연스러운 단어 시퀀스를 찾아내는 모델로, 보편적인 방법은 이전 단어들로 다음 단어를 예측하는 것이다.(양쪽단어로 중간을 추론하기도_BERT)
이때 언어모델이 주어진 단어로 아직 모르는 단어를 예측하는 작업을 언어 모델링(Language Modeling)이라고 한다.
 단어 시퀀스에서는 기계 번역(Machine Translation), 오타 교정(Spell Correction), 음성 인식(Speech Recognition)으로 적절한 문장을 판단한다.
기본적으로 사용되는 다음 단어 등장 확률은 conditional probability를 이용하여 표현하며, 앞의 단어들이 나온 후에, 나올 확률 즉 P(w5 | w1,w2,w3,w4)와 같이 표현한다.
 우리가 단어를 판단하는 것과 같이 기계도 앞의 단어를 고려하여 후보가 될 수 있는 단어들의 확률을 계산하고, 가장 높은 확률을 가진 단어를 채택한다. ex)검색어 자동 완성

    [통계적 언어 모델(Statictical Language Model, SLM)]
Conditional probability의 기본 속성을 이용하여 많은 확률에 대해 일반화를 해보면 P(x1,x2,x3...xn)=P(x1)*P(x2 | x1)*P(x3 | x1,x2)*...*P(xn | x1...xn-1)이 되며, 이를 chain rule이라고 한다.
즉, 여러 확률에 대하여 개별적인 조건부 확률들의 곱으로 나타낼 수 있다.
 위의 내용을 실제 'An adorable little boy is spreading smiles'에 적용시켜보면, P(An adorable little boy is spreading smiles)=P(An)*P(adorable | An)*...*P(smiles | An adorable little boy is spreading)이 된다.
즉, 문장 전체의 확률을 구하기 위하여 각 단어에 대한 예측 확률들을 곱하는 것이다.
 이를 실제로 언어모델 측면으로 생각해보면, 카운트에 기반하여 조건부 확률을 계산할 수 있다. 즉, 이전의 단어들이 등장한 횟수(count)중에 다음 단어가 등장한 횟수(count)의 비율로 나타내는 것이다.
이를 위의 예시에 적용해보면, P(is | An adorable little boy)=count(An adorable little boy is)/count(An adorable little boy)로 구한다는 의미이다.
 다만 위와 같이 다음 단어의 확률을 알기 위해 언어모델의 카운드 기반 접근에도 한계는 존재하는데, 학습된 corpus로 도출된 확률이 0인 경우 이를 정말
시퀀스의 개연성이 없어서 0인 것인지, 훈련 데이터가 적어서 count되지 않아 0인 것인지를 구분하기 어렵고(like 관성, 중력) 이를 희소 문제(Sparsity Problem)이라고 한다.
이러한 카운트기반접근의 한계를 해소하기 위해 n-gram 언어 모델이나 스무딩, 백오프같은 generalization기법이 존재하나, 근본적인 해결은 불가능하여 이를 통계적 언어모델이 아닌 인공 신경망 언어 모델로 넘어가게 된다.

    [N-gram 언어 모델(N-gram Language Model)]
SLM일종이나, 위에 설명한 모델과는 달리 이전에 등장한 모든 단어를 고려하지 않고 일부단어만 고려한다. 이때 확인하는 일부 단어의 개수가 n-gram에서의 n의 의미이다.
 n-gram은 n개의 연속적인 단어 나열을 의미하며, corpus에서 n개의 단어 뭉치 단위를 하나의 토큰으로 간주한다. 예시로 An adorable little boy is spreading smiles의 경우
unigrams(하나씩): an, adorable, ... / bigrams(두개씩): an adorable, adorable little, little boy, .../ trigrams(세개씩): an adorable little, adorable little boy, ... / 4-grams(4개씩): an adorable little boy, ...이 된다.
즉, 다음에 나올 단어의 예측을 오직 n-1개의 단어에만 의존하는 것이다. 즉, corpus에서 boy is spreading이 1000번 등장했고, 뒤에 insults온게 500개, smiles가 200개이면 딱 그 단어들에만 의존하기에 50%인 insults를 선택한다.
 하지만 N-gram모델은 당연히 전체 문장을 고려한 언어 모델보다는 문맥연결이 안되는 등 정확도가 떨어질 수 밖에 없다. n-gram이 일부단어를 보아 corpus에서 count될 확률을 높였지만 여전히 n희소 문제가 존재하며,
n을 선택하는 것인지는 trade-off(모순)이 발생하는데, n이 크면 n-gram카운트 확률이 적어져 희소문제가 심각해지며 모델 사이즈가 커지고, n이 작으면
count는 잘되지만 정확도는 현실의 확률분포와 멀어진다. 고로 적절한 n이 중요하며 2~5사이로 권장하고 있다.
 부가적으로 언어모델의 성능을 높이기 위해서는 적용 분야(Domain)에 맞는 corpus의 수집이 중요하다. 고로 제대로된 corpus이냐 아니냐에 따라서
언어 생성의 정확도가 비약적으로 달라지기에 corpus의 선정역시 중요하다. 적용 분야를 생각하자.
 위에 잠시 언급한 것 처럼 N-gram Language model, SLM등의 한계점을 극복하기 위해 분모분자에 수를 더해 0을 방지하는 generalization방법이 존재하지만, 근본적인 문제는 그대로라 NNBLM이 많이 사용된다.

    [한국어에서의 언어 모델(Language Model for Korean Sentences)]
한국어는 어순을 바꿔도 의미가 전달되어 다음 단어의 확률을 구하기 까다로우며, 교착어이기에 tokenization을 통해 접사나 조사를 분리해야하고, 띄어쓰기가 없어도
의미전달이 되기에 이를 체크하여 띄어쓰기가 되지 않은 데이터가 학습되어 언어 모델이 이상하게 동작하는 것을 방지할 필요가 있다.

    [펄플렉서티(Perplexity, PPL)
두개의 모델의 성능을 비교하는데에 실제 작업을 시키는 것보다 테스트 데이터에 대하여 모델내에서 자신의 성능을 수치화하는 펄플렉서티(perplecity)를 사용하는 것이 더 좋다.
 perplexity(헷갈리는)는 언어모델 평가지표로, 수치가 낮을수록 언어모델의 성능이 좋다는 것을 의미한다. PPL은 문장의 길이로 정규화된 문장 확률의 역수이다.
이때 PPL이 선택가능한 경우의 수를 분기계수(branching factor)라고 하는데, 특정 시점에 평균적으로 몇개의 선택지를 가지는지를 의미한다. ex) PPL(W)=10
(항상 그렇지만, 평가방법에 있어서 데이스 데이터 상에서 높은 정확도를 가진다는 것과 실제로 좋은 모델이라는 것은 같이 수반되지 않는다는 것을 유의해야 한다.)
"""
