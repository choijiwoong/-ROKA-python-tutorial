def outer():#실행시 inner함수가 정의되고, inner이라는 변수가 내부 inner함수 객체를 바인딩한다. 
    def inner():
        print("inner")
    return inner#마지막으로 inner가 바인딩하고있는 함수 객체를 리턴하고,

f=outer()#그 값을 global영역 f가 바인딩한다.
f()#고로 f는 outer()내부의 inner()함수를 바인딩하고 있기에 'inner'문자열이 정상적으로 출력된다.
#다시말해서, outer내부 inner이라는 이름이 inner()함수객체를 바인딩하고, global f가 마찬가지로 같은 inner()함수 객체를 바인딩한다.


def outer():
    inner=3
    return inner

f=outer()#outer를 호출하여 반환된 값이 f에 저장
print(f,'\n')#f는 지금 int


#Enclosed function locals
def outer():
    num=3
    def inner():
        print(num)
    return inner

f=outer()#outer는 inner()함수객체를 바인딩하는 inner를 반환하기에 마찬가지로 f는 inner()함수 객체를 바인딩한다.
f()#이때 inner()함수객체는 내부의 num을 LEGB규칙에 의거하여 탐색하는데, local에 없기에 Enclosed function area를 탐색하게 되고,
#inner()함수 객체를 감싸는 outer()함수 객체 영역 안에서 num변수를 찾아 print(num)명령에 사용한다.

#하지만 refcount관점에서 보면, f가 바인딩한 inner()은 refcount가 존재한다고 하지만, outer()은 호출과 동시에 refcount가 0이 되기에 GC에 의해 삭제되어
#num=3이 사라져야 정상이다. 하지만 정상적으로 3을 출력하는 것은, 내부 함수 객체가 생성될 때 Enclosed function locals영역의 변수를 자신의 객체 안에 저장해두기 때문이다.
print(f.__closure__[0].cell_contents)#f변수가 바인딩하는 function타입의 객체는 __closure__속성을 갖고 있는데, 튜플 타입의 객체를 바인딩한다.
#즉, enclosure function area의 변수를 tuple로 저장하기에 현재 __closure__이 가리키는 tuple의 크기는 1이고, [0]으로 접근 가능했던 것이다.
#이 tuple은 cell타입의 객체로 이루어졌는데(바인딩하는데) cell객체의 cell_contents속성에 해당 값이 저장되는 구조이다.

print(type(f.__closure__))#tuple
print(type(f.__closure__[0]))#cell
print(dir(f.__closure__[0]))#cell의 속성들 출력

def outer(num):
    def inner():
        print(num)
    return inner

f=outer(5)#inner()함수 객체를 바인딩하는데, num을 자신의 __closure__에 저장한다.(인자역시)
print(f())#고로 정상적으로 5출력

print(f.__closure__[0].cell_contents)#5
