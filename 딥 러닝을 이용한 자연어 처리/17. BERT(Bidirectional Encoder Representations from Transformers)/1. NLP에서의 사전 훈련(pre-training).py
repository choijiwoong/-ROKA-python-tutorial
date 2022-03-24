""" 트랜스포머 이후도, RNN계열의 LSTM, GRU가 대체되어가며 트랜스포머 계열의 BERT, GPT, T5등의 pre-trained language model이 등장했다.
BERT와 BERT에서 파생된 ALBERT(A Lite BERT Self-supervised Learning of Language Representations), RoBERTa(A Robustly Optimized BERT Pretraining Approach),
ELECTRA(Efficiently Learning an Encoder that Classifies Token Replacements Accurately)를 학습해보자.

    [NLP에서의 Pre-training]
 1. 사전 훈련된 워드 임베딩
Word2Vec, FastText, GloVe등 워드 임베딩 방법론들로 하여금 Embedding Layer을 학습시켜 사용하거나 사전 학습된 Embedding Vector들을 가져와 사용할 수 있었다.
이 방법의 단점은 하나의 단어가 하나의 벡터값으로 매핑되기에 다의어나 동음이의어를 구분하지 못했다.

 2. 사전 훈련된 언어모델
학습된 LSTM언어모델은 별도의 레이블이 부착되지 않은 텍스트 데이터로도 학습이 가능한데, 더 나아가 순방향 언어모델과 역방향 언어모델을 따로 학습시키는 ELMo가 있다.
순방향과 역방향을 분리시키지 않으면, 순방향에서 예측해야하는 단어를 이미 역방향에서 관측한 셈이 되어버린다.  이로서 Word2Vec, FastText, GloVe의 다의어구분 문제를 해결했다.(문맥을 고려하기에)
이에 그치지 않고 RNN계열의 고질적인 문제를 탈피하고자 시도한 것이 트랜스포머이며 트랜스포머의 디코더를 12개층쌓아 만든 것이 GPT-1 언어모델이다.
하지만 언어적으로 실제 문맥은 양방향이기에 분리형이 아닌 실제 양방향 구조를 도입하기 위해 등장한  새로운 구조의 언어 모델이 Masked Language Model이다.

 3. 마스크드 언어 모델(Masked Language Model)
입력 텍스트의 단어집합 15%를 랜덤으로 마스킹하고, 이를 예측하게 한다.(빈칸채우기 문제)

    [버트(Bidirectional Encoder Representations from Transformers, BERT)]
 1. BERT의 개요
레이블이 없는 방대한 데이터 훈련뿐만아닌, 파인튜닝(Fine-tuning)레이블있는 데이터 추가훈련을 통한 하이퍼파라미터 재조정)을 거쳤기 때문이다.
다르게 말하면, BERT위에 분류를 위한 신경망을 한 층 추가하여 BERT의 사전 지식을 분류와 같은 상황에서 이용하는 것이며, ELMo나 OpenAI GPT-1이 파이 튜닝 사례의 대표적인 예이다.

 2. BERT의 크기
BERT-Base는 트랜스포머 인코더 12개, BERT-Large는 24개를 쌓아 만들어졌으며, Base는 Layer_num=12, D_model=768, Aheads=12 & Large는 L=24, D=1024, A=16의 크기를 갖는다.
Base는 GPT-1과 비교를 위한 버전이기에 Hyperparameter가 동일하고, Large는 실제 사용하기 위한 최대 성능의 버전이다.

 3. BERT의 문맥을 반영한 임베딩(Contextual Embedding)
BERT는 ELMo나 GPT-1처럼 Contextual Embedding을 사용하기에, 연산을 거치면 문장의 문맥을 모두 참고한 임베딩이 된다.
트랜스포머 인코더를 쌓은 것이기에 각층마다 Multi-head Self-Attention과 Positional-wise FFNN을 수행하기 때문이다.

 4. BERT의 서브워드 토크나이저: WordPiece
단어보다 더 작은 단위로 쪼개며 동작방식은 다르지만 서브워드를 병합하여 Vocab을 만드는 과정이 Byte Pair Encoding(BPE)와 유사하다.
BERT 토큰화 수행 아이디어는 단어집합에 존재하지 않을 경우 서브워드로 분리하여 첫번째 서브워드를 제외한 나머지 앞에 ##를 붙여 토큰으로 사용한다(중간서브워드표시, 복구위함)"""
import pandas as pd
from transformers import BertTokenizer

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

try:
    print(tokenizer.vocab['embeddings'])#존재하지 않는 단어라 key error가 발생한다.
except KeyError:
    print('embeddings is not in vocab!: ')
    
result=tokenizer.tokenize('Here is the sentence I want embeddings for.')#vocab에 없는 단어는 서브워드분리
print('BertTokenizer result: ', result)#embeddings에 대하여 em, ##bed, ##ding, ##s로 분리되었는데 이 단어들은 모두 단어집합에 존재한다.


with open('vocabulary.txt', 'w') as f:#tokenizer vocab을 저장.
    for token in tokenizer.vocab.keys():
        f.write(token+'\n')
df=pd.read_fwf('vocabulary.txt', header=None)#읽어들이기
print('단어집합의 크기: ', len(df))
print('4667번 단어 확인: ', df.loc[4667].values[0])##ding
#참고로 BERT의 특별 토큰은 다음과 같다. [PAD]_0, [UNK]_100, [CLS]_101, [SEP]_102, [MASK]_103
"""
 5. 포지션 임베딩(Position Embedding)
프랜스포머의 사인&코사인 기반 PE함수를 이용하는 positional encoding과 유사한 Position Embedding을 사용한다.(학습을 통해 얻음)
위치 정보를 위한 임베딩층을 하나 더 사용하여 0~n-1번 포지션 임베딩 벡터를 각각 더해준다. 즉, BERT에서는 워드 임베딩, 포지션 임베딩 총 2개의 임베딩 층이 사용된다.
추가적으로 Segment Embedding이 더 사용되는데 나중에 나온다.

 6. BERT의 사전 훈련(Pre-training)
1) 마스크드 언어 모델(Masked Language Model, MLM)
15%의 단어중 80%는 [MASK], 10%은 랜덤으로 변경, 나머지는 그대로 해서 예측하게 한다. 파인 튜닝 단계에서는 [MASK]를 사용하지 않기에 이 차이를 완화하기 위함이다.
각각 해당하는 부분의 출력층 벡터만을 사용하며, 다른 위치에서의 예측은 손실함수에서 무시한다.

2) 다음 문장 예측(Next Sentence Prediction, NSP)
BERT는 두개의 문장이 이어지는 문장인지를 훈련시키는데, 이어질 경우 IsNextSentence, 반대의 경우 NotNextSentence로 Label을 매긴다.
BERT의 입력으로 넣을 때에는 문장의 끝에 [SEP]토큰을 넣고, 문장의 시작을 나타내는 [CLS]토큰의 출력층 위치에서 이진 분류 문제를 풀어 이를 확인한다.
NSP의 학습 이유는 QA(Question Answering), NLI(Natural Language Inference)와 같이 관계를 이해하는 것이 중요한 태스크가 있기 때문이다.

 7. 세그먼트 임베딩(Segment Embedding)
문장 구분을 위한 임베딩 레이어로, 첫번째 문장에는 Sentence 0 임베딩, 두번째 문장에는 Sentence 1임베딩을 더해주는 방식이며, 임베딩 벡터는 2개만 사용된다.
 결론적으로 BERT에는 WordPiece Embedding(실질적입력), Position Embedding(위치정보), Segment Embedding(두문장구분용) 총 3개의 임베딩이 사용된다.
[SEP]와 세그먼트 임베딩으로 구분되는 BERT의 두개의 문장은 두개의 문서일 수도 있으며, 한개의 문서내에서 분류하는 경우 전체 입력에 Sentence 0 임베딩만을 더해 하나의 문장을 입력받아 사용할 수도 있다.

 8. BERT를 파인 튜닝(Fine-tuning)하기
1) Single Text Classification
하나의 문서에 대한 텍스트 분류로 문서의 시작에 [CLS]토큰을 입력하여 이 출력을에 Denselayer같은 FC를 추가하여 분류예측을 한다
2) Tagging
각 토큰의 위치에 Dense를 사용하여 예측한다. 개체명 인식 혹은 품사태깅을 의미한다.
3) Text Pair Classification or Regression
텍스트쌍을 입력으로 받는 태스크의 경우 주 문장이 모순관계(Contradiction), 함의 관계(entailment)_A가 참이면 B도 참, 중립 관계(neutral)등
논리적 관계를 분류하는 Natural language inference에 사용될 수 있다.
 이러한 경우 텍스트를 쌍으로 받기에 그 사이에 [SEP]토큰을 넣고, Sentence 0 임베딩, Sentence 1 임베딩을 사용하여 문서를 구분한다.
4) 질의 응답(Question Answering)
또다른 텍스트쌍 입력 태스크로, 질문과 본문이라는 두 개의 텍스트 쌍을 입력하며 대표적인 데이터 셋으로 SQuAD(Stanford Question Answering Dataset) v1.1가 있다.
본문의 일부를 추출해서 질문에 답하는 것이다.

 9. 어텐션 마스크(Attention Mask)
추가적인 BERT 시퀀스 입력으로, 불필요한 패딩을 구분하는 입력으로 0이 마스킹하는 것을 의미한다."""
