"""
import pandas as pd
import pandas_profiling

#load file
data=pd.read_csv('archive/spam.csv', encoding='latin1')
print(data[:5])

#make report
pr=data.profile_report()#report after profiling save to pr
pr.to_file('./pr_report.html')#save pr to html

#check report
print(pr)
"""

#[Machin Learning Workflow]
#Acquisition(corpus_자연어데이터)->Inspection & exploration(Exploratory Data Analysis, EDA단계)->Preprocessing ans Cleaning
#->Modeling and Training->Evaluation->Deployment


#[Text preprocessing]_토큰화, 정제, 정규화_영어와 달리 한국어의 경우 단순히 구두점이나 특수문자를 정제하면 의미가 달라질 수 있기에 영어보다 복잡하다.
#영어의 경우도 아포스트로피 Don't에 관해 정제하는것에 대해서도 논의가 가능하지만, 이미 공개된 도구 NLTK를 사용하면 그 처리결과에 대해 확인할 수 있다.
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

#result of tokenize by using word_tokenize
print('단어 토큰화1: ', word_tokenize("Don't be folled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
#Don't->Do, n't, Jone's->Jone, 's

#result of tokenize by using wordPunctTokenizer_구두점을 별도로 분류하는 특성이 있다.
print('단어 토큰화2: ', WordPunctTokenizer().tokenize("Don't be folled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
#Don't->Do, n't, Jone's->Jone, ', s

#케라스또한 토큰화 도구로서 text_to_word_sequence를 지원한다._소문자로 바꾸면서 의미있는 구두점은 남겨둔다.
print('단어 토큰화3: ', text_to_word_sequence("Don't be folled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
#Don't->don't, Jone's->jone's


#[한국어의 토큰화 어려움]
#토큰화는 단순히 코퍼스에서 구두점을 제외하고 공백 기준으로 잘라내는 작업이 아니라 섬세한 알고리즘이 필요한데, 단순제외하면 .같이 문장의 경계를 알려주는 정보가 사라질 수 있고,
#ph.D, AT&T, 45.55$같이 의미가 있는 경우도 있고, what are을 줄이는 what're처럼 줄임말의 힌트가 될 수 있기때문이다.
 #고로 표준으로 사용되는 토큰화 방법은 Penn Treebank Tokenization규칙이다. 이 규칙은 하이푼으로 구성된 단어는 하나로 유지하며, doesn't처럼 아포스트로피로 접어가 함께하는 단어는 분리해준다.
from nltk.tokenize import TreebankWordTokenizer

tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print("트리뱅크 워드토크나이저: ", tokenizer.tokenize(text), end='\n\n')
#home-based, does, n't, own, .


#[Sentence Tokenization(sentence segmentation)]_토큰화 단위가 문장일 경우
#NLTK에서 영어 문장의 토큰화를 수행하는 sent_tokenize를 이용해보자
from nltk.tokenize import sent_tokenize

text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print("문장 토큰화1: ", sent_tokenize(text))#Cool

text="I am actively looking for Ph.D. students. and you are a Ph.D student."#.이 다수 등장하는 경우
print('문장 토큰화2: ', sent_tokenize(text))#Cool

#한국어 문장 토큰화의 경우 KSS(Korean Sentence Splitter)을 사용하면 좋다.
import kss
text='딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('한국어 문장 토큰화: ', kss.split_sentences(text), end='\n\n')#Cool...Maybe?


#[한국어 토큰화의 어려움]
"""
영어의 경우 합성어나 he's같은 줄임말만 고려해도 토큰화가 잘 작동하는데, 한국어의 경우 띄어쓰기 단위인 어절단위로 해야하기에 단어 토큰화와 별개의 토큰화가 필요하며,
한국어는 조사,어미 등을 붙여서 말을 만드는 교착어이기때문에 근본적으로 지양된다.
 교착어의 특성 상 같은 단어임에도 조사가 달라 다른 단어로 인식하면 자연어처리가 복잡해지기에 대부분의 한국어 NLP에는 조사를 분리한다.
고로 morpheme(형태소)를 반드시 이해해야하는데, 뜻을 가진 가장 작은 말의 단위인 자립형태소와 의존형태소로 분류한다. 자립형태소는 자립하여 사용할 수 있는 형태소로
그 자체로 단어가 되지만, 의존 형태소를 다른 형태소와 결합하여 사용되는 형태소이다. 형태소 분리의 예시로 '에디가 책을 읽었다'는 단어 토큰화로는 '에디가', '책을', '읽었다'이지만
형태소단위로 분해하면 자립형태소: 에디, 책 /의존형태소: -가, -을, 읽-, -었, -다 가 된다. 즉, 한국어는 단어토큰화와 유사한 형태를 얻기 위해 형태소 토큰화를 수행해야한다.
 또한 한국어는 띄어쓰기를 하지 않아도 의미를 할 수 있기에 띄어쓰기를 하지 않은 문장도 고려해야한다. 이는 고급진 언어적 차이 용어로 한국어는 모아쓰기 방식이고, 영어는 풀어쓰기 방식이라는 언어적 특성의 차이이다.
"""

#[품사 태깅(Part-of-speech tagging)]
"""
단어의 표기는 같지만 품사에 따라서 단어의 의미가 달라지기도 하기때문에 해당 단어가 어떤 품사로 쓰였는지 보는 것이 주요지표가 될 수 있다.
고로 단어 토큰화 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해놓는데, 이 작업을 품사 태깅(part-of-speech tagging)이라고 한다. NLTK과 KoNLPy에서 품사태킹을 지원한다.
"""

#[NLTK와 KoNLPy를 이용한 영어, 한국어 토큰화 실습]
from nltk.tokneize import word_tokenize
from nltk.tag import pos_tag

text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
tokenized_sentence=word_tokenize(text)

print('단어 토큰화: ', tokenized_sentence)
print('품사 태깅: ', pos_tag(tokenized_sentence))#Penn Treebank POS Tags에서 PRP_인칭대명사, VBP_동사, RB_부사, VBG_현재부사, IN_전치사, NNP_고유명사
#NNS_복수형 명사, CC_접속사, CT_관사를 의미한다.

#한국어 자연어 처리의 경우 KoNLPy를 이용하여 가능한데, 형태소 분석기로 Okt(Open Korean Text), 메캅(Mecab), 코모란(Komoran), 한나눔(Hannanum), 꼬꼬마(Kkma)가 있다.
#아래응 Okt와 Kkma를 이용한 morpheme tokenization예시이다.
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt=Okt()
kkma=kkma()

print('OKT 형태소 분석: ', okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))#motphs_형태소 추출
print('OKT 품사 태깅: ', okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))#pos(part_of_speech tagging)
print('OKT 명사 추출: ', okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))#명사추출
#OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
#OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
#OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
