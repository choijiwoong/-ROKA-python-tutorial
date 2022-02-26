"""Integer Encoding처럼 문자를 숫자로 바꾸는 여러 기법중 하나로, 기존의 vocabulary는 book과 books처럼 단어의 변형도 다른 언어로 간주했다.
문자에서 숫자로 변형하는 것에서 더 구체적으로 문자에서 벡터로 바꾸는 기법이 원-핫 인코딩이다. 단어집합의 크기를 벡터의 차원으로 하며, 표현하고 싶은 단어의
인덱스를 1, 다른 인덱스에 0을 부여하는 방식이다."""
#[원-핫 인코딩이란]
from konlpy.tag import Okt

okt=Okt()
tokens=okt.morphs("나는 자연어 처리를 배운다")#한국어 형태소 분석기
print("[원-핫 인코딩이란]\ntokens: ", tokens)

word_to_index={word: index for index, word in enumerate(tokens)}
print("단어 집합: ", word_to_index)

def one_hot_encoding(word, word_to_index):#단어와 word_to_index를 전달
    one_hot_vector=[0]*(len(word_to_index))
    index=word_to_index[word]
    one_hot_vector[index]=1
    return one_hot_vector

print(one_hot_encoding("자연어", word_to_index), end='\n\n\n')

#[케라스(Keras)를 이용한 원-핫 인코딩(One-Hot Encoding)]
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])#빈도수 기준 vocabulary생성
print("[케라스(keras)를 이용한 원-핫 인코딩(One-Hot Encoding)]\nvocabulary: ", tokenizer.word_index)

sub_text="점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded=tokenizer.texts_to_sequences([sub_text])[0]#Integer Encoding 테스트. [0]은 list로 변환되는데 sub_text가 이미 list여서 이중리스트 [[2,5,1,6,3,7]]어서지 별 의미 없음
print("encoded: ", encoded)

one_hot=to_categorical(encoded)
print("one-hot: ", one_hot, end='\n\n\n')

"""[원-핫 인코딩(One-Hot Encoding)의 한계]
벡터를 저장하기 위한 공간이 계속 늘어나며, 단어의 유사도를 표현하지 못한다. 이러한 단점을 해결하기 위해 잠재 의미를 반영하여
다차원 공간에 벡터화하는 기법으로 카운트 기반의 LSA(잠재의미분석),HAL이 있으며, 예측기반의 NNLM, RNNLM, Word2Vec, FastText등이 있다.
또한 카운트 기반과 예측 기반을 합친 GloVe도 있다."""
