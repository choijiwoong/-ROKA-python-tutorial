#함수 데코레이터
def inner():#메모리공간에 inner함수 객체가 존재하며
    print("inner function is called")#내용은 print를 담고있다.
inner()#inner함수객체를 호출
print()

def deco(f):#메모리 공간에 deco함수 객체가 존재하며
    def wrapper():#내부함수로 메모리공간에 wrapper함수 객체가 존재한다.
        print("-"*20)
        f()#wrapper내부 함수에선 외부 함수의 인자로 들어온 함수를 호출한다.
        print("-"*20)
    return wrapper#deco(외부함수)는 자신의 내부에서 정의된 함수객체를 리턴하는 closure구조이다. wrapper는 deco()안의 wrapper()를 binding하고 있다.
decorated_inner=deco(inner)#decorated_inner는 deco가 리턴하는 내부함수 객체를 바인딩하며, 별도의 __closure__속성에 deco(외부함수)의 인자를 저장하고 있다.
decorated_inner()#decorated_inner가 바인딩하고있는 wrapper()함수객체를 실행한다.


@deco#위와 같은 효과(deco라는 별도의 함수의 인자로 inner function을 넣어 내는 효과)를 wrapper느낌으로 할 함수의 이름을 @와 같이 쓰므로서 사용이 가능하다.
def inner():
    print("inner function is called")
inner()
