#[numpy로 패딩]
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

#지난번 데이터 활용
preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer=Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)#빈도수 기준으로 단어집합 형성
encoded=tokenizer.texts_to_sequences(preprocessed_sentences)#to integer
print("[numpy로 패딩]\nencoded: ", encoded)

max_len=max(len(item) for item in encoded)#encoded된 항목들의 길이 중 최댓값을 계산
print('최대 길이: ', max_len)

for sentence in encoded:#padding
    while len(sentence)<max_len:
        sentence.append(0)
padded_np=np.array(encoded)
print('padded_np: ', padded_np, end='\n\n\n')
#이처럼 특정 값을 채워 데이터의 크기(shape)를 조정하는 것을 패딩(padding)이라고 하며 0을 사용한 패딩을 특별히 zero padding이라고 한다.


#[케라스 전처리 도구로 패딩하기]
from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded=tokenizer.texts_to_sequences(preprocessed_sentences)
print("[케라스 전처리 도구로 패딩하기]\nencoded: ", encoded)

padded=pad_sequences(encoded)#기본설정은 문서 앞에 0을 채우는데, 만약 numpy 패딩처럼 앞을 0으로 채우고 싶다면 padding='post'를 인자로 넣으면 된다.
print("padded(pre): ", padded)

padded=pad_sequences(encoded, padding='post')
print("padded(post): ", padded)

print("numpy패딩과 keras post패딩 비교: ", (padded==padded_np).all(), end='\n\n')

#maxlen인자로 최대 길이 제한(평균길이 20, 하나만 5000일때 5000으로 padding하지 않아도 되는 상황 등)
padded=pad_sequences(encoded, padding='post', maxlen=5)
print("padded with maxlen=5", padded, end='\n\n')

#이러한 경우 초과되는 앞의 데이터가 손실되는데, 만약 앞을 남겨두고 뒤를 손실시키고 싶다면 truncating='post'를 인자로 주면 된다.
padded=pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
print("padded with truncating='post': ", padded, end='\n\n')

#만약 0이 아닌 다른 숫자로 패딩하고 싶다면 value=last_value를 인자로 주면 된다.
last_value=len(tokenizer.word_index)+1#예시로 단어집합의크기보다 1큰 숫자를 padding value로 사용
print("last_value: ", last_value)
padded=pad_sequences(encoded, padding='post', value=last_value)
print("padded with value=last_value: ", padded, end='\n\n')
