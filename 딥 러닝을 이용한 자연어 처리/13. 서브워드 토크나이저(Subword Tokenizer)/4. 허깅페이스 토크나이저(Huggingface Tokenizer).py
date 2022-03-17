    #[1. BERT의 워드피스 토크나이저(BertWordPieceTokenizer)]_구글이 공재한 BERT의 WordPiece Tokenizer을 허깅페이스 스타트업에서 직접 구현한 패키지
import pandas as pd
import urllib.request
from tokenizers import BertWordPieceTokenizer
#데이터로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
#전처리하여 txt저장
naver_df=pd.read_table('ratings.txt')
naver_df=naver_df.dropna(how='any')
with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))

#버트워드피스토크나이저
tokenizer=BertWordPieceTokenizer(lowercase=False)#대소구분O, 악센트 제거X(ô → o)_trip_accents인자가 왜 없징..사라졌낭...오픈소스의 폐해..

#데이터 학습
data_file='naver_review.txt'
vocab_size=30000
limit_alphabet=6000
min_frequency=5

#인자: limit_alphabet: 병합 전의 초기 토큰의 허용 개수?, min_frequency: 이 이상은 되어야 병합시켜줌
tokenizer.train(files=data_file, vocab_size=vocab_size, limit_alphabet=limit_alphabet, min_frequency=min_frequency)
tokenizer.save_model('./')#학습된 vocab저장

#load & use
df=pd.read_fwf('vocab.txt', header=None)#고정너비형식의 행 표(총 30000개의 단어) (수직데이터)

encoded=tokenizer.encode('아 배고픈데 짬뽕이 먹고싶다')
print('토큰화 결과: ', encoded.tokens)#호오 토큰화 결과과 굉장히 굉장히 흥미로운데 배고픈을 글자수 살려서 배고, ##픈으로 토큰화했네.
print('정수인코딩: ', encoded.ids)
print('디코딩: ', tokenizer.decode(encoded.ids))

encoded=tokenizer.encode('모닝똥 대신 모닝커피 마시며 여유를 즐기려는데 비가와~!')
print('토큰화 결과: ', encoded.tokens)#다른예로 해보니 글자수를 살린건 아닌듯. 보편적인 글자수로 #하거나 #을 그냥 2개 고정적으로 사용하는듯. 별로 안흥미로운걸로. 다만 단어와 접사는 구분잘될듯
print('정수인코딩: ', encoded.ids)
print('디코딩: ', tokenizer.decode(encoded.ids))

    """[2. 기타 토크나이저]
ByteLevelBPETokenizer(BERT에사용), CharBPETokenizer(오리지널 BPE), SentencePieceBPETokenizer(센텐스핏와 호환되는 BPE구현체), ByteLevelBPETokenizer(BPE의 바이트레벨버전)등이 존재한다."""
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceBPETokenizer

tokenizer=SentencePieceBPETokenizer()
tokenizer.train('naver_review.txt', vocab_size=10000, min_frequency=5)

encoded=tokenizer.encode('이 영화는 정말 군대같습니다')
print(encoded.tokens)
