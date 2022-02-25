#[PyKoSpacing]
#유용한 한국어 전처리 패키지 중 하나이다. 대용량 corpus를 학습하여 만들어진 띄어쓰기 딥 러닝 모델이다. 띄어쓰기를 대신 해줌.

sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'
new_sent=sent.replace(" ", '')#임의로 띄어쓰기가 업는 문장을 만들고
print('[PyKoSpacing]\n띄어쓰기 없는 sent(new_sent): ', new_sent, end='\n\n')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#인클루드에러 방지 To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
from pykospacing import Spacing

spacing=Spacing()
kospacing_sent=spacing(new_sent)

print("kospacing_sent: ", kospacing_sent, end='\n\n\n')


#[Py-Hanspell]_네이버 맞춤법 검사기를 바탕으로 만들어진 패키지이다.
from hanspell import spell_checker

sent="맞춤법 틀리면 외 안되? 쓰고싶은대로쓰면돼지 "
spelled_sent=spell_checker.check(sent)#vocabulary, calculated time등의 정보가 포함된 객체 생성.
print("[Py-Hanspell]\n기존의 문장: ", sent, "\n\nspelled_sent(spell_checker가 반환한 객체 정보 확인): ", spelled_sent, end='\n\n')

hanspell_sent=spelled_sent.checked#그 객체 내용중에 checked내용만 가져오기
print("hanspell(checked내용만 가져오기): ", hanspell_sent, end='\n\n')

#띄어쓰기 또한 보장한다. 다른 모델이기에 PyKoSpacing과는 조금 다르다.
spelled_sent=spell_checker.check(new_sent)#띄어쓰기 안된 문장(이전에 PyKoSpacing에서 활용한 변수)으로 object생성
hanspell_sent=spelled_sent.checked
print("\n기존의 문장: ", new_sent, "\n\nhanspell을 이용하여 맞춤법 보정한 문장: ", hanspell_sent, end='\n\n\n[SOYNLP]\n')


#[SOYNLP를 이용한 단어 토큰화]_품사태깅, 단어 토큰화 등을 지원하는 단어 토크나이저로, 비지도 학습으로 토큰화한다.
#이 토크나이저는 내부적으로 응집확률(cohesion probability)와 브랜칭 엔트로피(branching entropy)를 활용하는 단어점수표로 동작한다.
#기존의 형태소 분석기는 신조어같은 형태소분석기에 등록되지 않은 단어의 경우 제대로 구분을 못하였다.

 #신조어 문제
from konlpy.tag import Okt
tokenizer=Okt()
print("Okt(기존의 형태소 분석기): ", tokenizer.morphs('에이비식스 이대휘 1월 최애돌 기부 요정'))

#이를 해결하기 위해 특정시퀀스가 자주 등장하는 빈도가 높고, 앞뒤로 완전히 다른 단어가 등장하는 것을 고려하여 최애돌같은 문자 시퀀스를 형태소로 판단하는 토크나이저가 soynlp이다.

 #학습하기
import urllib.request
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor

#학습에 필요한 한국어 문서 다운
#urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")

#훈련 데이터를 다수의 문서로 분리
corpus=DoublespaceLineCorpus("2016-10-20.txt")
len(corpus)

#상위 3개의 문서만 출력
i=0
for document in corpus:
    if len(document)>0:
        print(document)
        i+=1
    if i==3:
        break

#soynlp는 학습 기반의 단어 토크나이저이므로 KoNLPy에서 제공하는 형태소 분석기들과는 달리 학습과정을 거쳐야 한다.
#전체 코퍼스로부터 응집확률과 브랜칭엔트로피 단어점수표를 계산한다.
word_extractor=WordExtractor()#instantiation
word_extractor.train(corpus)#train with corpus
word_score_table=word_extractor.extract()#extract word_score_table

 #SOYNLP의 응집 확률(cohesion probability)_얼마나 뭉쳐서 같이 등장하는지. 갑자기 낮아지면 그 전꺼가 적ㅡ절.
"""이는 substring이 얼마나 응집하여 자주 등장하는지를 판단하는 척도로 문자열을 문자단위로 분리하여 내부 문자열을 만드는 과정에서
왼쪽부터 순서대로 문자를 추가하며 각 무낮열이 주어졌을 때 그 다음 문자가 나올 확률을 계산하여 누적곱을 한 값이다."""
print('"여의도"의 응집확률: ', word_score_table['여의도'].cohesion_forward)#more high!
print('"여의도한"의 응집확률: ', word_score_table['여의도한'].cohesion_forward)
print('"여의도한강"의 응집확률: ', word_score_table['여의도한강'].cohesion_forward)
print('"여의도한강공"의 응집확률: ', word_score_table['여의도한강공'].cohesion_forward)
print('"여의도한강공원"의 응집확률: ', word_score_table['여의도한강공원'].cohesion_forward)
print('"여의도한강공원에"의 응집확률: ', word_score_table['여의도한강공원에'].cohesion_forward)
#훠훠..위의것보다 낮아져서 적합한게 여의도한강공원이라하려했는데 응집확률이 여의도한강공원보다 여의도한강공원에가 더 높누ㅋㅋ 무튼 이해는 갔고 완벽은 없다는 교휸..

 #SOYNLP의 브랜칭 엔트로피(brancing entropy)_다음문자예측 엔트로피. 갑자기 늘어나면 지금께 적ㅡ절
"""확률분포의 엔트로피로 다음 문자가 등장할 수 있는지를 판단할 수 있는 척도이다. 주어진 문자 시퀀스에서 다음 문자 예측을 위해 헷갈리는 정도로 비유가 가능하다.
예측 불정확도로 볼 수 있어, 완성된 단어에 가까워질수록 점점 줄어드는 양상을 보인다."""
print('"대한"의 브랜칭 엔트로피: ',word_score_table["대한"].right_branching_entropy)#줄
print('"대한민"의 브랜칭 엔트로피: ',word_score_table["대한민"].right_branching_entropy)#어
print('"대한민국"의 브랜칭 엔트로피: ',word_score_table["대한민국"].right_branching_entropy)#든..#늘어났다! 조사의 가능성 때문이다. 이 단어가 적합하다!
#역시나 훠훠...대한민의 엔트로피가 0이 되어버리네.. 예시인데는 이유가 있고만...약간 정확하지 않다기보다 학습을 위한 최선의 결과를 보여주는고만..

 #SOYNLP의 L tokenizer_어절 토큰은 주로 L 토큰+R 토큰(공원+에)로 나뉘는데 이 토크나이저는 L+R로 나누되, 분리 기준을 점수가 가장 높은 L 토큰을 찾아낸다.
from soynlp.tokenizer import LTokenizer

scores={word:score.cohesion_forward for word, score in word_score_table.items()}#word_score_table의 아이템을 word:score의 응집확률 dictionary로.
l_tokenizer=LTokenizer(scores=scores)
print("L토크나이저로 L토큰과 R토큰 분리: ", l_tokenizer.tokenize("국제사회와 우리의 노력들로 범죄를 척결하자", flatten=False))

 #최대 점수 토크나이저_띄어쓰기안된문장에서점수높은글자시퀀스순차적으로찾기
from soynlp.tokenizer import MaxScoreTokenizer

maxscore_tokenizer=MaxScoreTokenizer(scores=scores)
print("최대점수토크나이저로 점수높은글자 순차찾기: ", maxscore_tokenizer.tokenize("국제사회와우리의노력들로범죄를척결하자"), end='\n\n\n[SOYNLP를 이용한 반복되는 문자 정제]\n')#정확하누!


#[SOYNLP를 이용한 반복되는 문자 정제]_ㅋㅋㅋㅋㅋㅋㅋ ㅎㅎ같이 반복되는 것은 하나로 정규화
from soynlp.normalizer import *

print(emoticon_normalize('아 시이이이이이이이이발 졸려ㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗㅗ', num_repeats=2))
print(emoticon_normalize('사이사이사이사이사이다! 우리이런사이다', num_repeats=2), end='\n\n\n[Customized KoNLPy]\n')

#[Customized KoNLPy]
"""만약 한국어 형태소 분석기를 사용하여 단어 토큰화를 했는데, 고유명사와 같은 부분에서 이상하게 나뉘어 진다면
직접 사용자 사전을 형태소 분석기에 추가해줄 수 있다."""
from konlpy.tag._okt import Twitter#customized_konlpy에서 Twitter형태소 분석기를 제공한다. 이게 가장 쉬운 사용자패키지이다.
#UserWarning: "Twitter" has changed to "Okt" since KoNLPy v0.4.5.래서 import Okt해도 안되는뎀..
twitter=Twitter()

print(twitter.morphs('은경이는 사무실로 갔습니다.'))

twitter.add_dictionary('은경이', 'Noun')#직접 단어 추가.
print(twitter.morphs('은경이는 사무실로 갔습니다.'))
