def hello():
    print("hello")
f=hello
print(id(hello))#함수 객체가 할당된 주소 출력. 함수 이름은 함수 객체 자체를 바인딩한다. f와 hello는 함수 객체를 가리킬 뿐이다.


class Func:
    def __call__(self):
        print('호출 됨')

f=Func()
f()
