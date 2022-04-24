#Local(함수안), Enclosed function locals(내부함수에서 자신의 외부함수 범위), Global, Built-in(내장함수)순으로 탐색한다(like namespace)
a=10

def test():
    a=20
    print(a)
    
test()

#
a=10#Global!
#No outside function!
def test():#No Local!
    print(a)

test()

#
a=10

def test():
    a=20
    print(a)

test()
print(a)#cannot access local a in test function!


#번외로 함수 내에서 Global변수의 값을 수정하려면_global
a=10

def test():
    global a#전역변수 a를 사용할거에요
    a=20

test()
print(a)
