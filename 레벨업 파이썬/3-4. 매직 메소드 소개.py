#__method__식을 magic method 혹은 special method라고 부른다.
class Car:
    def __init__(self):
        print("complete")


def func():
    print("hello")
func()

class MyFunc:
    def __call__(self, *argc, **kwards):
        print("called")
f=MyFunc()
f()#함수도 객체이다.  파이썬의 함수를 function클래스의 객체이며, 함수의 이름은 해당 클래스의 객체를 바인딩하는 변수이다.


#객체 바인딩 시 .은 __getattribute__를 호출한다.
class Stock:
    def __getattribute__(self, item):
        print(item, "객체에 접근")

s=Stock()
s.data#.data는 __getattribute__(self, data)를 호출시켜준다.
