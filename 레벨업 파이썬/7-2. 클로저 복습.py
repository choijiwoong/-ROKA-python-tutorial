#데코레이터 사용 예
class Car:
    def __init__(self, model):
        self.model=model

    def get_model(self):
        return self.model

c=Car("GV80")
print(c.get_model())#객체에 저장된 속성을 얻기 위해서 메서드를 호출하였다.
print(c.model)


class Car:
    def __init__(self, model):
        self.model=model

    @property
    def get_model(self):
        return self.model

c=Car("GV80")
print(c.get_model)#get_model()을 호출하여 사용하는 것이 아닌 get_model 함수 객체가 마치 Car 클래스의 Property인 것 처럼 사용이 가능하다.(변수처럼)

print()
#클로저 복습_외부 함수 안에 내부 함수를 정의하고 정의된 내부 함수를 리턴하는 구조이다.
def outer(out1):
    def inner(in1):
        print("inner function called")
        print("outer argument: ", out1)
        print("inner argument: ", in1)
    return inner#정의된 내부함수를 리턴

f=outer(1)#f는 inner함수 객체를 바인딩하는데, __closure__속성에 호출에 필요한 외부 함수의 내용인 out1변수를 저장하고 있다.
f(10)#클로저에서 외부 함수의 인자로 함수를 넘겨줄 수도 있다.(똑같이 바인딩하는 것이기에)

print()
def outer(out1):
    def inner(in1):
        print('inner function called')
        print('inner argument: ', in1)
        out1()
    return inner

def hello():
    print("hello")

f=outer(hello)#outer의 인자로 functor를 전달해도 뭐 없다. 마찬가지로 저장할 뿐.
f(10)
