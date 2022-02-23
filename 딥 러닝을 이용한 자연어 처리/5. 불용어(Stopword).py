#유의미한 단어 토큰만을 선별하는 과정에서 큰 의미가 없는 I, my, me등의 단어는 불용어라고 하며 NLTK에서는 100여개 이상의 불용어가 정의되어 있다.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

#[불용어 확인하기] 
stop_words_list=stopwords.words('english')#english 불용어를 리스트로 가져온다
print('불용어 개수: ', len(stop_words_list))
print('불용어 10개 출력: ', stop_words_list[:10], end='\n\n')

#[불용어 제거하기]
example="Family is not an important thing. It's everthing."
stop_words=set(stopwords.words('english'))#as set

word_tokens=word_tokenize(example)#tokenize example

result=[]
for word in word_tokens:
    if word not in stop_words:#불용어 set에 없는 것만 append, 불용어는 생략
        result.append(word)
print('불용어 제거 전(example): ', word_tokens)
print('불용어 제거 후(example): ', result, end='\n\n')#굉장히 신기하게 의미추론이 되네.

#[한국어에서 불용어 제거하기]
okt=Okt()#한국어 형태소분석기

example="고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨데 삼겹살을 구울 때는 중요한 게 있지."
stop_words="를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"#임의로 한국어 불용어 set을 만들어서 테스트.

stop_words=set(stop_words.split(' '))#split하여 한국어 불용어 셋 생성
word_tokens=okt.morphs(example)#형태소 분석기 Okt로 형태소단위 split

result=[word for word in word_tokens if not word in stop_words]#불용어 확인 by comprehension

print('불용어 제거 전(example): ', word_tokens)
print('불용어 제거 후(example): ', result)
