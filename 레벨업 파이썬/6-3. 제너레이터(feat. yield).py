#yield는 코드가 중지된 상태를 저장해 둘 수 있다.
def num_gen():
    for i in range(3):
        yield i

g=num_gen()#제너레이터 객체 생성

num1=next(g)#next로 접근
num2=next(g)#코드를 이어서 실행할 수 있다!
num3=next(g)
#num4=next(g)#StopIteration!

print(num1, num2, num3,'\n')


g=num_gen()

for i in g:
    print(i)#StopIteration을 마지막에 반환하기에 iterable처럼 for을 사용할 수 있다.


#제너레이터는 큰 규모의 확장성이 있는 프로그램을 개발하기 위해 사용된다.
def 빵만들기(n):
    빵쟁반=[]
    for i in range(n):
        빵="빵"+str(i)
        빵쟁반.append(빵)
    return 빵쟁반

def 빵포장(빵):
    print("{} 포장완료.".format(빵))

for i in 빵만들기(100):
    빵포장(i)


def 빵만들기(n):#제너레이터를 사용하지 않은 위의 코드와 비교했을 때, 빵을 별도의 공간으로 모을 필요가 없다는 장점이 있다. (한번에 리턴이 아닌 바로 리턴하기에 저장공간이 불필요하다)
    for i in range(n):
        빵="빵"+str(i)
        yield 빵

def 빵포장(빵):
    print("{} 포장완료 ".format(빵))

for i in 빵만들기(100):
    빵포장(i)
