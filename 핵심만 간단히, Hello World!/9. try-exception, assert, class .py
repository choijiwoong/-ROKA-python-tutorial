#자신의 모듈을 설치된 패키지처럼 사용하고싶다면 sys.path에 append로 경로를 추가해주면 된다.

#[1. exception]_try-except구조
try:
    print("must execute")
    4/0
    print("not printed")
except (ZeroDivisionError, IndexError) as e:
    print(e)
except:
    print("annonymous error occur!")
else:
    print("Cool! work well")
finally:
    print("This sentence is printed by finally!")


#[2. assert]
def test(t):
    assert type(t) is int, "not integer!!!!"

lists=[1,3,6,2,3,4,7,1,8,67,10]
for i in lists:
    test(i)


#[3. class]
class temp:
    a=None

    def __init__(self, str):
        self.a=str

    def driving(self, str):
        print("we are driving as %s"%(str))
tory=temp("solid")
print(tory.a)
tory.driving("seoul")


class Car:
    def __init__(self, num):
        self.horpower=num

    def get_horsepower(self):
        return self.horpower
print(Car.get_horsepower)#unbounded method yet
#print(Car.get_horsepower())#not proper argument(not inistantiation yet)
print(Car.get_horsepower(Car(120)))#work well!
print(Car(120).get_horsepower())#work well!
print(Car(120).get_horsepower)#bounded method now!

#한번 바운드 되면 인자를 전달하지 않아도 된다.
a=Car(120).get_horsepower#이미 Car(120)의 get_horsepower함수이기에
print(a())#인자없이 ()로 호출
