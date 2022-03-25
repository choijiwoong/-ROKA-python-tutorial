    #[구글 BERT의 Next Sentence Prediction]
 #1. Next Sentence Prediction & Tokenizer
import tensorflow as tf
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer

model=TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased')

 #2. Input of BERT
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "pizza is eaten with the use of a knife and fork. In casual settings, however, it is cut into wedges to be eaten while held in the hand."

encoding=tokenizer(prompt, next_sentence, return_tensors='tf')#두개의 문장을 Integer Encoding
print('prompt와 next_sentence의 정수 인코딩: ', encoding['input_ids'])#101과 102는 [CLS], [SEP]토큰이다.
print('tokenizer.cls_token: ', tokenizer.cls_token,'(',tokenizer.cls_token_id,')')#[CLS] ( 101 )
print('tokenizer.cls_token: ', tokenizer.sep_token,'(',tokenizer.sep_token_id,')')#[SEP] ( 102 )
print('정수인코딩의 디코딩결과(입력의 구성 확인): ', tokenizer.decode(encoding['input_ids'][0]))#두개의 문장 앞에 [CLS], 문장마무리마다 [SEP]
print('세그먼트 인코딩 결과: ', encoding['token_type_ids'])#Sentence 0, Sentence 1의 세그먼트 인코딩 결과 확인 by 0 & 1

 #3. 다음 문장 예측하기
logits=model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]#model에 인코딩입력을 넣으면, softmax이전의 값인 logits을 리턴한다.(token_type_ids로 segment encoding전달)
softmax=tf.keras.layers.Softmax()
probs=softmax(logits)
print('예측값: ', probs)
print('최종 예측 레이블: ', tf.math.argmax(probs, axis=-1).numpy())#0. BERT는 이어질 경우 0 이어지지 않는 경우를 1로 Binary Classification한다.

#이번에는 상관없는 2개의 문장이다
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."#피자
next_sentence = "The sky is blue due to the shorter wavelength of blue light."#하늘

encoding=tokenizer(prompt, next_sentence, return_tensors='tf')#두 문장을 tokenizer 사용하여 인코딩한다.

logits=model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]#입력의 정수 인코딩과 세그먼트 인코딩을 모델에 전달하여 logits값 get
softmax=tf.keras.layers.Softmax()
probs=softmax(logits)#Functional API꼴로 Softmax 통과
print('최종 예측 레이블: ', tf.math.argmax(probs, axis=-1).numpy())#1


    #[한국어 BERT의 Next Sentence Prediction]
 #1. Next Sentence Predictiojn & Tokenizer
import tensorflow as tf
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer

model=TFBertForNextSentencePrediction.from_pretrained('klue/bert-base', from_pt=True)#TFBertNextSentencePrediction으로 klue/bert-base로드
tokenizer=AutoTokenizer.from_pretrained('klue/bert-base')

 #2. Next Sentence Prediction
#이어지는 경우
prompt = "2002년 월드컵 축구대회는 일본과 공동으로 개최되었던 세계적인 큰 잔치입니다."
next_sentence = "여행을 가보니 한국의 2002년 월드컵 축구대회의 준비는 완벽했습니다."

encoding=tokenizer(prompt, next_sentence, return_tensors='tf')

logits=model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]#Next Sentence Prediction의 logit값 get
softmax=tf.keras.layers.Softmax()
probs=softmax(logits)
print('최종 예측 레이블: ', tf.math.argmax(probs, axis=-1).numpy())#0(Numpy로 바꾸는 이유는 출력 이쁘게하려고)

#이어지지 않는 경우
prompt='2022년 대통령 선거는 이전과는 대비됩니다.'
next_sentence='우리 부대 새로운 대령이 임관되었습니다.'

encoding=tokenizer(prompt, next_sentence, return_tensors='tf')

logits=model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
softmax=tf.keras.layers.Softmax()
probs=softmax(logits)
print('최종 예측 레이블: ', tf.math.argmax(probs, axis=-1).numpy())#1
