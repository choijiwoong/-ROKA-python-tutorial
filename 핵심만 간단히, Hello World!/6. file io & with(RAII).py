#[1. basic file io]
"""
#text.txt
Hello World!
Good Morning
Good Afternoon!
Good Evening!
"""

#file read
f=open("test.txt", 'r')
#first way for get all contents
while True:
    l=f.readline()
    if not l:
        break
    print(l)
    
#second way for get all contents
print(f.readlines())#read file by format of list

#third way for read all contents
print(f.read())

f.close()


#file write
f=open("test2.txt", 'w')
f.write("Good Afternoon!")
f.write(["Many line!\n", "Maybe line.\n"])
f.close()

#[2. Context Managers(with statement)]_RAII를 적용하기 위해 close필요없기끝 with문을 사용하여 다음과 같이 연다_like try-with-resources in JAVA
#이러한 것은 with문이 실행하는 class의 enter메서드와 exit메서드가 정의되어 있기 때문이다. like begin() & end() on range-based loop in C++
with open('text.txt', 'r') as f:
    f.readline()#scope limit
#tensorflow도 세션을 열고 닫을 때 위와같은 RAII를 사용하기 위해 아래처럼 사용한다.
#with th.Session() as sess:
    #blah blah..

#enter와 exit를 정의한 class로 with에서 작동하게 하는 간단한 예시는 아래와 같다.
class test:
    def __enter__(self):
        print("enter")
        a='hello'
        return a
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")#print hello
with test() as t:
    print(t)
