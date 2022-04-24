a=1,2
b=(1,2)
print(a, type(a))#tuple? tuple! 괄호없이 콤마로만 구분해도 튜플로 인식하며, 이를 튜플 패킹이라고 부른다.
print(b, type(b))

data=(1,2,3)
n1, n2, n3=data#튜플 언패킹
print(n1, n2, n3)

n1=data[0]
n2=data[1]
n3=data[2]#튜플 언패킹은 이러한 부담을 줄여준다.


scores=(1,2,3,4,5,6)
low, *others, high=scores#튜플 언패킹 시 튜플의 일부 값을 하나의 변수로 묶어서 바인딩이 가능하다. by *(가변)
print(others)


def foo():
    return 1, 2, 3#함수는 하나의 값만 리턴할 수 있는데, 여러 값을 리턴하면 튜플로 패킹된 후 튜플 객체가 리턴된다.

val=foo()
print(type(val))


#언패킹과 함수 호출
def hap(num1, num2, num3, num4):
    return num1+num2+num3+num4

scores=(1,2,3,4)
result=hap(scores[0], scores[1], scores[2], scores[3])#이러한 귀찮음을 아래와 같이 해소 가능하다.
print(result)

scores=(1,2,3,4)
result=hap(*scores)#tuple unpacking! 각 매개변수로 인자가 전달된다.
print(result)
