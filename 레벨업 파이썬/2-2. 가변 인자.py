#Positional variable argument(*args)
def foo(*args):#여러 입력에 대하여 튜플로 저장한 뒤 이 튜플 객체를 바인딩한다.
    print(args)

foo(1,2,3)
foo(1,2,3,4)

#Keyword variable arguments(**kwargs) 키를 가진 인자들!
def foo(**kwargs):
    print(kwargs)

foo(a=1, b=2, c=3)#{'a': 1, 'b': 2, 'c': 3}

#같이 사용하기
def len(*argc, **kwargs):
    pass

def foo(*args, **kwargs):
    print(args)
    print(kwargs)

foo(1, 2, 3, a=1, b=1, c=2)

#함수 인자로 리스트/튜플 전달하기
def foo(a, b, c):
    print(a, b, c)

data=[1,2,3]
foo(data[0], data[1], data[2])

foo(*data)#same result

#함수를 호출할 때 **의 의미
def foo(**kwargs):
    print(kwargs)

foo(a=1, b=2)#직접 key와 value를 매칭해서 전달할 수도 있고

params={'a': 1, 'b':2}
foo(**params)#**로 kwarg형식으로 전달 가능.
