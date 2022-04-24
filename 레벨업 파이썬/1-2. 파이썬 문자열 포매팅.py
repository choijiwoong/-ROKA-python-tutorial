#문자열 포맷팅
name="minsu"
score=90
print("%s의 점수는 %d점입니다."%(name, score))#c언어 유사 포맷
print("{}의 점수는 {}점입니다.".format(name, score))#python3이상에서 지원하는 문자열 클래스의 format메서드 사용
print(f"{name}의 점수는 {score}점입니다.")#python3.6부터 지원하는 f-string사용

#특수한 글자 출력
data=3
fmt="{{ {} }}".format(data)#중괄호 출력을 원할 경우 두개를 적어준다.
print(fmt)#{ 3 }

data=3
fmt=f"{{ {data} }}"
print(fmt)#f-string도 동일하다

#자리수 채우기
a=3
mystr=f"{a:02d}"#a를 2자리 정수로 출력!
print(mystr)#03

#실수 다루기
a=3.141592
mystr=f"{a:.2f}"#a를 .2까지 출력!
print(mystr)
