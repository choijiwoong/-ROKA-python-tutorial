#[collections.Counter]
from collections import Counter

#Integer Encoding의 과정과 모듈사용의 방법을 분리하기 위해 별도의 py로 정리중이라, 이전 py의 raw_text와 preprocessed_sentences의 값만 별도로 표기해두었다.
raw_text="A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
preprocessed_sentences=[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
print("[collections.Counter]\npreprocessed_sentences: ", preprocessed_sentences, end='\n\n')

all_words_list=sum(preprocessed_sentences, [])#words=np.hstack(preprocessed_sentences)와 동일. vocabulary생성을 위해 모든 단어를 합친다.
print(all_words_list, end='\n\n')#(약간 flatten과도 비슷한듯 )

vocab=Counter(all_words_list)#collections에서 지원해주는 Counter이용하여 빈도수 directory생성
print("vocabylary: ", vocab, end='\n\n')#(당연하게도 빈도수 1도 저장)
print("frequency of barber: ", vocab['barber'], end='\n\n')

#most_common을 이용하여 상위 빈도수를 가진 단어를 인자 사이즈만큼 filtering할 수 있다.
vocab_size=5
vocab=vocab.most_common(vocab_size)
print("vocab with most_common method: ", vocab, end='\n\n')#second로는 빈도수를 포함하고 있다.(not rank yet)

word_to_index={}#빈도수에 따라 랭킹리스트 생성
i=0
for (word, frequency) in vocab:
    i=i+1
    word_to_index[word]=i#vocab의 순서대로 indexing(이미 빈도수대로 정렬됨)
print("word_to_index with collections.Counter & most_common(Interget Encoding의 기준): ", word_to_index, end='\n\n\n')


#[NLTK의 FreqDist]
from nltk import FreqDist
import numpy as np

vocab=FreqDist(np.hstack(preprocessed_sentences))#Counter()과 똑같은 빈도수 계산 도구 FreqDist(). pre-sen를 합친다음에 빈도수계산
print("[nltk.FreqDist]\nvocab: ", vocab, end='\n\n')
print("frequency of barber: ", vocab['barber'], end='\n\n')

vocab_size=5
vocab=vocab.most_common(vocab_size)#위의 collections.Counter예시와 마찬가지로 상위를 거르는데 most_common사용.
print("vocab with most_common method: ", vocab, end='\n\n')

word_to_index={}
i=0
for (word, frequency) in vocab:
    i=i+1
    word_to_index[word]=i
print("word_to_index with nltk.FreqDist & most_common(Integer Encoding의 기준)", word_to_index, end='\n\n\n')


#[enumerate]
vocab_size=5
vocab=Counter(sum(preprocessed_sentences, [])).most_common(vocab_size)#vocab엔 상위 count정보 존재.
word_to_index={word[0]: index+1 for index, word in enumerate(vocab)}#dictionary. vocab의 word값: index+1(on enumerate)로 word_to_index생성

print("[enumerate]\nword_to_index with enumerate & it's index(Integer Encoding의 기준): ", word_to_index, end='\n\n')


#[enumerate의 간단한 이해]
test_input=['a', 'b', 'c', 'd', 'e']
for index, value in enumerate(test_input):
    print("value: {}, index: {}".format(value, index))#(index, value)를 반환하며, 0부터 시작한다.
print("\n\n[tensorflow.keras.preprocessing.text.Tokenizer]")


#[Keras의 텍스트 전처리]
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()#instantiation
tokenizer.fit_on_texts(preprocessed_sentences)#fit_on_texts는 빈도수를 기준으로 단어집합을 생성. (중복제거)
print("word_index(tokenizer.word_index): ", tokenizer.word_index, end='\n\n')
print("word_counts(tokenizer.word_counts): ", tokenizer.word_counts, end='\n\n')
print("convert corpus to predecided index(text_to_sequences method): ", tokenizer.texts_to_sequences(preprocessed_sentences), end='\n\n')#아마 이미 빈도수 기준으로 된듯

#빈도수가 가장 높은 단어n개만을 위해 most_common을 사용했는데, 이를 tokenizer=Tokenizer(num_words=숫자)로 대체가능하다.
vocab_size=5
tokenizer=Tokenizer(num_words=vocab_size+1)#tokenizer가 출력한 위의 word_index를 보면 0을 사용하지 않지만, 사용하는 것처럼 vocabsize+1을 하는 이유는 padding때문이기에 0도 단어집합의 크기로 고려해야한다.
tokenizer.fit_on_texts(preprocessed_sentences)
print("word_index after Tokenizer(num_words=vocab_size+1): ", tokenizer.word_index, end='\n\n')#빈도수적용X
print("word_counts after Tokenizer: ", tokenizer.word_counts, end='\n\n')#빈도수적용X
print("tokenizer.texts_to_sequences(preprocessed_sentences) after Tokenizer: ", tokenizer.texts_to_sequences(preprocessed_sentences), end='\n\n')#이제야 빈도수 적용. 실제 적용은 texts_to_sequences사용시.

"""#(만약 word_index와 word_counts에서도 빈도수가 반영되게 하고 싶다면 아래의 코드를 참고)
print("참고1")
tokenizer=Tokenizer()#without num_words
tokenizer.fit_on_texts(preprocessed_sentences)

vocab_size=5
words_frequency=[word for word, index in tokenizer.word_index.items() if index>=vocab_size+1]#items의 index가 해당 사이즈를 초과한다면(알아서 빈도수대로 정렬되어있기에) 저장

for word in words_frequency:
    del tokenizer.word_index[word]#초과하는 항목을 word_index와 word_count에서 직접 제거
    del tokenizer.word_counts[word]
print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))"""

"""#(Keras의 Tokenizer는 OOV는 Integer Encoding과정에서 아예 단어를 제거하는데, 보존하고 싶다면 Tokenizer의 인자 oov_token을 설정하면 된다.)
print("참고2")
vocab_size=5
tokenizer=Tokenizer(num_words=vocab_size+2, oov_token='OOV')
tokenizer.fit_on_texts(preprocessed_sentences)

print("단어 OOV의 인덱스: {}".format(tokenizer.word_index['OOV']))
print(tokenizer.texts_to_sequences(preprocessed_sentences))"""
