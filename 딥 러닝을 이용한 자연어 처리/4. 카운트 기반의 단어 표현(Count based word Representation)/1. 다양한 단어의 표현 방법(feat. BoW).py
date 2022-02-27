"""
    [다양한 단어의 표현 방법]
크게 국소 표현(Local Representation_해당단어만 봄(뉘앙스X))과 분산 표현(Distributed Representation_주변을 참고(뉘앙스X))방법으로 나뉘며,
국소표현을 Discrete Representation으로, 분산 표현을 Continuous Representation이라고 부르기도 한다.
우리는 국소표현(Local Representation)에 속하며, COunt하여 수치화하는 Bag of Words에 대해 배울 예정이다.

    [Bag of Words(BoW)]
단어들의 순서상관없이, 오로지 출현 빈도(frequency)에만 집중하는 수치화 표현방법으로, 단어집합생성 후 해당 위치에 frequency를 기록한 벡터를 만들어 사용한다.
"""
#문서1: 정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.
from konlpy.tag import Okt

okt=Okt()

def build_bag_of_words(document):#문서 온점제거 및 형태소 분석을 통해 word_to_index(단어의 인덱스정보)와 bow(인덱스의 빈도수)반환
    document=document.replace('.', '')#온점 제거 (나중에 불용어인 조사를 제거하면 추가적인 정제가 가능하다.)
    tokenized_document=okt.morphs(document)#형태소 분석

    word_to_index={}#[word]:index
    bow=[]#frquency on that index

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word]=len(word_to_index)#다음 index에 새로 추가
            bow.insert(len(word_to_index)-1,1)#기본값 1
        else:
            index=word_to_index.get(word)
            bow[index]=bow[index]+1

    return word_to_index, bow#아직까지는 왜 word: frequency로 안만드는지 잘 모름. 많이 복잡해지려나

doc1="정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
vocab, bow=build_bag_of_words(doc1)#인덱스가 0부터 시작됨을 주의
print('[self]\nvocabulary(document1): ', vocab)
print('bag of words vector(document1): ', bow, end='\n\n')

#문서2: 소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.
doc2='소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'

vocab, bow=build_bag_of_words(doc2)
print('vocabulary(document2): ', vocab)
print('bag of words vector(document2): ', bow, end='\n\n')

#문서3: 문서1+문서2
doc3=doc1+' '+doc2
vocab, bow=build_bag_of_words(doc3)
print('vocabulary(document3): ', vocab)
print('bag of words vector(document3): ', bow, end='\n\n\n')


"""종종 여러 문서의 vocabulary를 합친 후 이를 이용하여 각각의 BoW를 구하기도 한다. 이러한 frequency기반 수치화는 문서분류나 문서간의 유사도를 구하는데에 용이하다.
BoW는 sklearn의 CountVectorizer으로 보다 편리하게 만들 수 있다. 다만 CountVectorizer는 단지 띄어쓰기만을 기준으로 BoW를 만들기에 한국어에 적용하면 제대로 BoW가 만들어지지 않는다."""
from sklearn.feature_extraction.text import CountVectorizer

corpus=['you know I want your love. because I love you.']
vector=CountVectorizer()

print('[CountVectorizer]\nbag of words vector(CountVectorizer): ', vector.fit_transform(corpus).toarray())#그냥 method외워서 사용해야할듯. vector출력 별거 안뜨네..
print('vocabulary(CountVectorizer): ', vector.vocabulary_, end='\n\n\n')#I는 자동적으로 길이가 짧아 없앴다.(전처리 of 정제)


"""BoW를 사용한다는 것은 frquency를 통해 단어의 중요도를 보겠다는 거기에 BoW생성과정에서 불용어를 제거하면 정확도를 높일 수 있다.
영어의 경우 CountVectorizer를 통한 불용어제거기능을 지원해준다."""
from sklearn.feature_extraction.text import CountVectorizer#그냥 다시 한번 임포트..
from nltk.corpus import stopwords

#불용어 사용자 정의 ver
text=["Family is not an important thing. It's everything."]
vect=CountVectorizer(stop_words=["the", "a", "an", "is", "not"])#stop_words리스트를 CountVectorizer생성시 argument로.
print('[stop_word]\nbag of words vector(with user-designed stop_word): ', vect.fit_transform(text).toarray())
print('vocabulary(with user-designed stop_word): ', vect.vocabulary_,end='\n\n')

#불용어 CountVectorizer 제공 ver
vect=CountVectorizer(stop_words="english")
print('bag of words vector(with default stop_word for english): ', vect.fit_transform(text).toarray())
print('vocabulary(with default stop_word for english): ', vect.vocabulary_,end='\n\n')

#불용어 NLTK 제공 ver
stop_words=stopwords.words('english')#nltk.corpus의 stopwords english버전을 가져와서
vect=CountVectorizer(stop_words=stop_words)#CountVectorizer의 stop_words인자에 설정.
print('bag of words vector: ', vect.fit_transform(text).toarray())
print('vocabulary: ', vect.vocabulary_)
