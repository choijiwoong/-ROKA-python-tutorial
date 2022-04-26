#가장 간단한 형태의 클로저
def outer(num):
    def inner():
        print(num)
    return inner#inner()함수객체에는 enclosed function local영역의 num값이 __closure__속성에 저장된다.

f1=outer(3)
f2=outer(4)#각각 자신의 함수 객체에 저장하기에 당연히 개별적인값이 출력된다.
f1()#3
f2()#4

#클래스를 이용한 클로저의 구현..이라는데 약간 개념만 좀 더 이해하라고 액션 흉내느낌으로 말한듯. 이게 뭔 클로저야 __closure__도 없는데...
class Outer:
    def __init__(self, num):
        self.num=num

    def __call__(self):
        print(self.num)

f1=Outer(3)
f1()
