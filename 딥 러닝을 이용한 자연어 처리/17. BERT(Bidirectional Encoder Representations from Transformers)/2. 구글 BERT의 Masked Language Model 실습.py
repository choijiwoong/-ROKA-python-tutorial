""" [구글 BERT의 Masked Language Model 실습]
 1. Masked Language Model & Tokenizer
BERT는 이미 학습된 모델이므로, 토크나이저는 BERT의 토크나이저를 사용하는 매핑 관계여야 한다."""
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer

model=TFBertForMaskedLM.from_pretrained('bert-large-uncased')#TFBertForMaskedLM으로 모델이름을 넣으면, [MASK]된 언어모델링 구조로 BERT를 로드한다.
tokenizer=AutoTokenizer.from_pretrained('bert-large-uncsed')#해당 모델이 학습되었을 당시의 토크나이저 로드

 #2. BERT의 입력
inputs=tokenizer('Soccer is a really fun [MASK].', return_tensors='tf')#문장을 MLM에 넣어 [MASK]단어 예측시키기 위한 정수인코딩
print('정수 인코딩 결과: ', inputs['input_ids'])
print('세그먼트 인코딩 결과(문장을 구분): ', inputs['token_type_ids'])#현재 입력이 문장 하나라 Sentence 0 즉 문장 길이 만큼의 0 시퀀스를 반환한다.
print('어텐션 마스크(실제 단어와 패딩 토큰 구분용): ', inputs['attention_mask'])#패딩이 없기에 문장길이만큼의 1을 얻는다.

 #3. [MASK] 토큰 예측하기
from transformers import FillMaskPipeline

pip=FillMaskPipeline(model=model, tokenizer=tokenizer)#마스크드 언어 모델의 예측 결과를 정리해서 보여준다.

pip('Soccer is a really fun [MASK].')#상위 5개 후보 단어를 반환한다.
pip('The Avengers is a really fun [MASK].')
pip('Sunday morning is good [MASK].')

"""
    [한국어 BERT의 Masked Language Model 실습]
blue/bert-base는 대표적인 한국어 BERT이다. 고로 TFBertForMaskedLM과 AutoTokenizer에 모델이름을 매칭 시켜 사용해보자.
 1. Masked Language Model & Tokenizer"""
from transformers import TFBertForMaskedLM
from transformers import AutoTokenizer

model=TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)#from_pt는 pytorch로 학습된 모델을 가져온다는 의미이다.(Korean LangUagE)
tokenizer=AutoTokenizer.from_pretrained('klue/bert-base')

 #2. BERT의 입력
inputs=tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='tf')
print('정수 인코딩: ', inputs['input_ids'])
print('세그먼트 인코딩: ', inputs['token_type_ids'])
print('어텐션 마스크: ', inputs['attention_mask'])

 #3. [MASK] 토큰 예측하기
from transformers import FillMaskPipeline

pip=FillMaskPipeline(model=model, tokenizer=tokenizer)

pip('축구는 정말 재미있는 [MASK]다.')
pip('군대는 [MASK]다.')#군대는 아니다ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 3순위
