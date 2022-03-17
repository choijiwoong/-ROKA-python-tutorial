""" 텐서플로우에서 제공하는 서브워드 토크나이저로, BPE와 유사한 Wordpiece Model을 채택한 모델이다."""
    #[1. IMDB 리뷰 토큰화하기]
import pandas as pd
import urllib.request
import tensorflow_datasets as tfds

urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
train_df=pd.read_csv("IMDb_Reviews.csv")#review열에 토큰화를 수행할 데이터가 저장되어있다.

tokenizer=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_df['review'], target_vocab_size=2**13)#인자로 data를 넣어 vocab을 생성하고, 고유 정수를 부여한다.
print('토큰화 된 서브워드들: ', tokenizer.subwords[:100],'\n')

print('비교를 위한 원본 데이터: ', train_df['review'][20])
print('Integer encoding: ', tokenizer.encode(train_df['review'][20]),'\n\n')#encode()를 이용해 Integer encoding result를 확인할 수 있다.

#encode와 decode의 테스트용 샘플 데이터
sample_string = "It's mind-blowing to me that this film was even made."
tokenized_string=tokenizer.encode(sample_string)#integer encoding
print('정수 인코딩 한 후의 문장: ', tokenized_string)
original_string=tokenizer.decode(tokenized_string)
print('디코딩한 후의 문장: ', original_string,'\n')

print('단어 집합의 크기: ', tokenizer.vocab_size)#8268
for ts in tokenized_string:#tokenized_string에 매칭된 정수를 확인(test sample)
    print(ts,'----->', tokenizer.decode([ts]))

#기존의 예제 문장에서 even이라는 단어에 임의로 xyz를 추가하여 없는 단어 문장으로 만든 후 인식시켜보자.
sample_string = "It's mind-blowing to me that this film was evenxyz made."

tokenized_string=tokenizer.encode(sample_string)#integer encoding결과를 tokenized_string에 저장
print('\n\n(xyz)정수 인코딩 후의 문장: ', tokenized_string)#evenxyz를 even x y z 로 분리하여 encode!

original_string=tokenizer.decode(tokenized_string)
print('다시 디코딩한 문장: ', original_string)#복원률을 확인(Well-done!)
for ts in tokenized_string:#각 token별 매핑된 integer value확인
    print(ts, '----->', tokenizer.decode([ts]))

    #[2. 네이버 영화 리뷰 토큰화하기]
import pandas as pd
import urllib.request
import tensorflow_datasets as tfds

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
train_data = pd.read_table('ratings_train.txt')

train_data=train_data.dropna(how='any')#null제거

tokenizer=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_data['document'], target_vocab_size=2**13)
print('\n\nsubword접근: ', tokenizer.subwords[:100])

#테스트
sample_string = train_data['document'][21]

tokenized_string=tokenizer.encode(sample_string)#네이버 데이터 기반으로 sample_string을 encode
print('(test)정수 인코딩 후의 문장: ', tokenized_string)
original_string=tokenizer.decode(tokenized_string)
print('(test)decode한 후의 문장: ', original_string)
for ts in tokenized_string:#매핑된 정수들 체크
    print(ts, '---->', tokenizer.decode([ts]))
