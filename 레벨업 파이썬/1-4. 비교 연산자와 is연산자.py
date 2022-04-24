#값은 ==, id(동일객체)는 is를 사용한다.
a=1000
b=1000
print(a==b)#True
print(a is b)#False..여야하는데..? 파이썬은 정수 256까지 해당 값이 존재하면 기존의 객체를 바인딩하게 한다는데 모든 수로 바뀌었나보네
print(id(a), id(b))

a=3
b=3
print(id(a), id(b))
print(a==b)#True
print(a is b)#True
