class Car:
    def __init__(self):
        self.wheels=4
        print("fuck")
        self.wheels=self.wheels+1
        print(self.wheels)

    def drive(self):
        print("drive")

mycar=Car()

print(hasattr(mycar, "wheels"))#hasattr은 해당 객체에 속성이 있는지를 bool로 알려준다.
print(hasattr(mycar, "drive"))

getattr(mycar, "wheels")

method=getattr(mycar, "drive")#get을 이용하면 멤버 메서드를 가져올 수 있다.(속성)
method()

method=getattr(mycar, "__init__")#오우 생성자도 되네...이게 모노..
print(method() ,type(method()))#타입은 None인데 실행은 잘되노..
