"""정규표현식 모듈에서 지원하는 함수로 re.compile()_정규표현식을 컴파일하여 패턴이 빈번한 경우 속도와 편의성 면 유리,
re.search()_문자열 전체가 정규표현식과 매치되는지, re.match()_문자열의 처음이 정규표현식과 매치되는지, re.split()_정규표현시 기준으로 분리하여 리스트리턴
re.findall()_문자열에서 정규표현식과 매치되는 모든 경우를 찾아 리스트리턴, re.finditer()_findall을 이터레이터리턴, re.sub()_문자열에서 정규표현식과 일치하면 다른문자열로 대체
참고로 match된다면 Match object를 리턴한다."""

import re

#.
r=re.compile("a.c")#a와 c사이에 문자 하나 있던거 같은데..
print(r.search("kkk"))
print(r.search("aic"), end='\n\n')

#?
r=re.compile("ab?c")#b가 있었나? 없었나? 상관없어~
print(r.search("abbc"))
print(r.search("ac"))
print(r.search("ac"), end='\n\n')

#*
r=re.compile("ab*c")#a와 c사이에 b가 없었나...있었나...많이 있었나... b*n은 0도 되고 inf도 되니..
print(r.search("a"))
print(r.search("ac"))
print(r.search("abbbbc"), end='\n\n')

#+
r=re.compile("ab+c")#b*와 비슷한데 최소 1개 이상..더한다면 0은 될 수 없지..
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbbc"), end='\n\n')

#^
r=re.compile("^ab")#^땅! ab로 시작한다
print(r.search("bbc"))
print(r.search("zab"))
print("abzsjljf", end='\n\n')

#{숫자}
r=re.compile("ab{2}c")#b가 2개인데 bb라고 쓰기 귀차낭
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbbbc"))
print(r.search("abbc"), end='\n\n')

#{숫자, 숫자}
r=re.compile("ab{2,8}c")#b가 2개~8개 범위 내라면 상관없어~
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbbbbbbbc"))
print(r.search("abbc"))
print(r.search("abbbbbbbbc"), end='\n\n')

#{숫자,}
r=re.compile("a{2,}bc")#a가 2개 이상이기만 하면 상관없어~
print(r.search("bc"))
print(r.search("aa"))
print(r.search("aabc"))
print(r.search("aaaaaaaabc"), end='\n\n')

#[]
r=re.compile("[abc]")#a,b,c중에 하나의 문자와 매치가 될거야.([a-z], [A-Z], [0-5]처럼 범위 지정도 가능)
print(r.search("zzz"))
print(r.search('a'))
print(r.search("aaaaaaa"))
print(r.search("baac"), end='\n\n')

r=re.compile("[a-z]")
print(r.search("AAA"))
print(r.search("111"))
print(r.search("aBC"), end='\n\n')

#[^문자]
r=re.compile("[^abc]")#a,b,c는 있으면 안돼!
print(r.search("a"))
print(r.search("ab"))
print(r.search("b"))
print(r.search("d"))
print(r.search("1"), end='\n\n')

#re.match()는 문자열의 첫 부분이 정규표현식과 매칭되는지를, re.search()는 전체 문자열에서 매칭되는것이 있는지를 비교한다. search는 정말 찾는거 match는 다짜고짜 바로 regex판단
r=re.compile("ab.")
print(r.match("kkkabc"))#k? 컷
print(r.search("kkkabc"))#abc? OK!
print(r.match("abckkk"), end='\n\n')#a? bc?!  OK!

#re.split()는 솔직히 다 알잖앙
text="사과 딸기 수박 메론 바나나"
print(re.split(" ", text))#공백을 기준으로 문자열 분리

text="""사과
딸기
수박
메론
바나나"""#오 이게 주석인줄만 알았는데 호오 약간 C++의 R""같은 느낌인가보네 다만 주석으로도 쓰이고
print(re.split("\n", text))

text="사과+딸기+수박+메론+바나나"
print(re.split("\+", text), end='\n\n')#특수 기호이기 때문에 \사용

#re.findall()_2nd argument에서 1nd argument로 건네진 regex에 매칭되는 모든 것들을 리스트로 반환
text="""이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""
print(re.findall("\d+", text))#숫자가 1개이상인거를 text전체에서 매칭된다면 리스트로 반환
print(re.findall("\d+", "문자열인데 숫자없는!!!"), end='\n\n')#Return empty list

#re.sub()_(regex, char_will_be_used_for_replacing, data). data의 regex매칭되는 모든걸 2nd arg로 대체.
text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."

preprocessed_text=re.sub('[^a-zA-Z]', ' ', text)#영어가 아닌 것들을 공백으로 치환(기호, 숫자...등등)
print(preprocessed_text, end='\n\n')

#정규 표현식 텍스트 전처리 예제
text="""100 John   PROF
101 James   STUD
102 Max   STUD"""
print(re.split('\s+', text))#최소 공백1개이상 기준으로 split
print(re.findall('\d+', text))#숫자 최소 1개이상 모두찾아 리스트로
print(re.findall('[A-Z]', text))#대문자로 이루어진거 모두 찾기
print(re.findall('[A-Z]{4}', text))#대문자가 연속으로 4개로 이루어진거 모두 찾기
print(re.findall('[A-Z][a-z]+', text), end='\n\n')#대문자+소문자최소 1개이상으로 이루어진거 모두 찾기

#정규 표현식을 이용한 토큰화_nltk.tokenize의 RegexpTokenizer
from nltk.tokenize import RegexpTokenizer

text="Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop"

tokenizer1=RegexpTokenizer("[\w]+")#영어,숫자로 이루어진거 하나 이상 기준으로 tokenizer
tokenizer2=RegexpTokenizer("\s+", gaps=True)#공백이 최소 하나 이상인거 기준으로 tokenizer. gaps는 해당 정규표현식을 기준으로 매칭되는것을 뽑는게 아니라
#해당 regex가 tokenizer하는 기준으로 사용됨을 의미한다.

print(tokenizer1.tokenize(text))
print(tokenizer2.tokenize(text))
