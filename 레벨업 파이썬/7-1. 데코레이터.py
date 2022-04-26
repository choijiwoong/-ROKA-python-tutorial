#데코레이터
def hello():
    print("hello")

hello()


def deco(fn):#입력으로 함수를 받는다.
    def deco_hello():
        print("*"*20)
        fn()
        print("*"*20)
    return deco_hello#내장 함수 deco_hello를 반환
hello=deco(hello)#데코레이트를 이용해 hello를 바꿔버림으로서 나머지 hello에서도 데코헬로 사용이 가능함.
hello()#function->decorator->decorated function


#@기호 사용하기: 단순히 어떤 함수에 기능을 추가하고자 하면 해당 함수 위에 @데코레이터함수 처럼 사용하면 된다.
@deco#hello2를 deco의 입력으로 받아 별이 추가된 deco_hello를 hello2에 저장.
def hello2():
    print("hello 2")

hello2()
