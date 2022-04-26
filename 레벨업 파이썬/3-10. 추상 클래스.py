from abc import *

class Car(metaclass=ABCMeta):
    @abstractmethod#이를 상속받는 모든 클래스에서 drive()의 재정의를 강제한다.
    def drive(self):
        pass

class K5(Car):
    pass#drive() is not defined!

#k5=K5()#TypeError

class K5(Car):
    def drive(self):
        print("k5 drive")
k5=K5()#well done
k5.drive()
