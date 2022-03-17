""" BPE, Unigram Language Model Tokenizer를 포함한 기타 서브워드 토크나이징 알고리즘들을 내장한 센텐스피스는 pretokenization없이
전처리가 안된 데이터(raw data)에 바로 적용할 수 있다. 센텐스피스는 사전 토큰화 작업없이 단어 분리 토큰화를 수행하기에 언어에 종속되지 않는다"""
    #[2. IMDB 리뷰 토큰화하기]
import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

 #1. 데이터 쑤까불리엣
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
train_df=pd.read_csv("IMDb_Reviews.csv")
print('리뷰 개수: ', len(train_df))

with open('imdb_review.txt', 'w', encoding='utf8') as f:#DataFrame내의 csv파일을 txt파일로 저장 for 센텐스피스 입력
    f.write('\n'.join(train_df['review']))
#센텐스피스로 단어 집합과 각 단어에 고유한 정수 부여
sp=smp.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
#학습시킬 파일, 모델이름, 단어집합크기, 사용할 모델(unigram(default), bpe, char, word), 문장의 최대 길이
#그 외에 인자로 pad_id, pad_piece, unk_id, bos_id, eos_id, user_defined_symbols...

vocab_list=pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
print('단어 집합의 크기: ', len(vocab_list))#위에 vocab_size 5000으로 제한함.

 #2. 모델 로드
sp=spm.SentencePieceProcessor()#Instantiation
vocab_file='imdb.model'
sp.load(vocab_file)#센텐스피스 모델 로드

lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]
for line in lines:#문장을 서브워드 시퀀스로 변환하는 encode_as_pieces, 문장을 정수 시퀀스로 변환하는 encode_as_ids도구 테스트.
    print(line)
    print(sp.encode_as_pieces(line))
    print(sp.encode_as_ids(line))
    print()

#센텐스피스의 여러 기능들
print('단어 집합의 크기: ', sp.GetPieceSize())#5000
print('정수로부터 매핑되는 서브워드 변환(int_to_subword): ', sp.IdToPiece(430))#_character
print('서브워드로부터 매핑되는 정수 변환(subword_to_int): ', sp.PieceToId('▁character'))#430(_아님.)
print('정수시퀀스를 문장으로 변환: ', sp.DecodeIds([41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]))#I have waited a long time for someone to film
print('서브워드시퀀스를 문장으로 변환: ', sp.DecodePieces(['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film']))#I have waited a long time for someone to film
print('문장을 정수시퀀스 혹은 서브워드 시퀀스로 변환(by argument): ', sp.encode('I have waited a long time for someone to film', out_type=str))#['▁I', '▁have', '▁wa', 'ited', '▁a', '▁long', '▁time', '▁for', '▁someone', '▁to', '▁film']
print('문장을 정수시퀀스 혹은 서브워드 시퀀스로 변환(by argument): ', sp.encode('I have waited a long time for someone to film', out_type=int))#[41, 141, 1364, 1120, 4, 666, 285, 92, 1078, 33, 91]

    #[3. 네이버 영화 리뷰 토큰화하기]_위와 동일한 과정
import pandas as pd
import sentencepiece as spm
import urllib.request
import csv

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
naver_df=pd.read_table('ratings.txt')

print('\n\nnull값체크: ', naver_df.isnull().values.any())#True
naver_df=naver_df.dropna(how='any')
print('null값체크 after dropna: ', naver_df.isnull().values.any())#False
print('리뷰 개수: ', len(naver_df))#199992

with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))#for 센텐스피스
spm.SentencePieceTrainer.Train('--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')#위의 데이터 이용 센텐스피스의 단어집합 생성
#SentencePiece의 vocab생성이 완료되면 naver.model과 naver.vocab 파일 2개가 생성되며, .vocab에서 학습된 subwords확인이 가능하다.

vocab_list=pd.read_csv('naver.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
print('vocab_list의 크기: ', len(vocab_list))#5000

sp=spm.SentencePieceProcessor()
vocab_file='naver.model'
sp.load(vocab_file)

lines = [
  "뭐 이딴 것도 영화냐.",
  "진짜 최고의 영화입니다 ㅋㅋ",
]
for line in lines:#서브워드 시퀀스로encode_as_pieces, 정수로 encode_as_ids
  print(line)
  print(sp.encode_as_pieces(line))
  print(sp.encode_as_ids(line))
  print()

#여러가지 기능 다시
print('단어집합 크기: ', sp.GetPieceSize())
print('정수->서브워드: ', sp.IdToPiece(4))
print('서브워드->정수: ', sp.PieceToId('영화'))
print('정수시퀀스->문장: ', sp.DecodeIds([54, 200, 821, 85]))
print('서브워드시퀀스->문장: ', sp.DecodePieces(['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']))
print('문장->정수시퀀스: ', sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=str))
print('문장->서브워드시퀀스: ', sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=int))

"""음 일단 특정 data를 사용하는 tokenizer을 바로 만들 수 있다는게 큰 장점인 것 같고,
사용법은 우선 spm.SentencePieceTrainer.Train을 이용하여 txt데이터로 vocab을 만든다.
그 뒤 sp=spm.SentencePieceProcessor()센튼스피스프로세서를 만든 뒤, 위의 과정에서 생성된 vocab파일을 로드시킨다.
그 뒤 encode_as_pieces, encode_as_ids, GetPieceSize, IdToPiece등 기능들을 사용하면 된다. 주된 기능은 서브워드시퀀스와 정수시퀀스, 문장간의 변환이 자유롭다는 것이다"""
