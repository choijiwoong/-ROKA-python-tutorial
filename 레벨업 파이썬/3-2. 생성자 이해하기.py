#생성자란?
class Person:
    def __init__(self):
        print("borned!")
p=Person()

#생성자의 위치: 클래스 공간(모든 객체에 의해 참조, 사용, 호출 될 수 있어야 하기에)

#인스턴스 개수 세기
class MyClass:
    count=0#공유 변수

    def __init__(self):
        MyClass.count+=1#직접 클래스 이름 공간을 호출한다.

    def get_count(self):
        return MyClass.count

a=MyClass()
b=MyClass()
c=MyClass()

print(a.get_count())#인스턴스의 메서드를 활용하여 접근
print(MyClass.count)#직접 접근
