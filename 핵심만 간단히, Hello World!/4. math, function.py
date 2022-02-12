#애초에 내용도 간략하기도 하고 나도 리마인드 목적이라 연습삼아 몇개 해보는 느낌으로 하자. not 정리

print(bin(10))#2진수변환

a=5
b=5
print(a is b)#identity operator. 같은 메모리 주소를 가졌는지를 테스트
print(5 is 5.0)#파이썬은 -5~256까지의 정수객체를 미리 만들어둔다.

#일반적인 변수 구조는 immutable로 독립되지만, 리스트이 경우 대입연산자로 shallow copy되기에 서로 영향을 미치는데, 이를 mutable이라한다

1 in [1,2]#와 같은 in & not in 연산자를 membership operator라고 한다.

#logical operators는 &&, ||이 아닌 글자 그대로 and, or이라고 사용한다.

#문자열은 리스트로 여겨지기에 인덱싱이 가능하다


#[function]
def add(x, y):#In python, function is object.
    return x+y
print('\n',add(3,5))

l=[3,4,add]
print(l[2](1,5))#special case of calling function

def ah(x):
    def ah2(y):
        return x+y
    return ah2
print(ah(100)(2))#very very special case of calling function

#[loop]
a=list(range(10))#make list by using range
for i in a:
    print(i)

words=[('apple', 'good'), ('banana', 'bad'), ('orange', 'excellent')]
for (word1, word2) in words:
    print(word1+"'s state is "+word2)
