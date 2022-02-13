#[1. inheritance]
class Car:
    kind="Sedan"
    countOfDoors=5
    hoursepower=0

    def __init__(self, str):
        self.name=str

    def sethorsepower(self, str):
        self.horsepower=str

    def driving(self, str):
        print("driving as %s"%(str))

class Car_nextGeneration(Car):
    autoPilot=True
    countOfAirbag=10

    def setwheelSize(self, str):
        self.wheelSize=str

    def driving(self, str):#method overriding
        print("auto driving as %s"%(str))

    def turnning(self, num):#new method
        self.turnninghorsepower=self.horsepower+num
        return self.turnninghorsepower
toch=Car("touch")#set name(member)
toch.horsepower=120
toch.driving("seoul")#print str

Mbbang=Car_nextGeneration("Mbbang")#set name(inherited member)
Mbbang.horsepower=300
Mbbang.wheelSize=20
print(Mbbang.turnning(200))
Mbbang.driving("Incheon")


#[2. Setter & Getter]
#initial problem of hiding
class CCar:
    def __init__(self, t):
        self.horsepower=t

    def gethorsepower(self):
        return self.horsepower

    def sethorsepower(self, str):
        self.horsepower=str
        
#solution for hiding
class Sol_CCar:
    def __init__(self, power=100):
        self.__horsepower=power
        print("make Sol_CCar object that's horsepower: ",power)

    def GetHorsepower(self):
        return self.__horsepower

    def __SetHorsepower(self, power):
        self.__horsepower=power

    def SetAndGet(self, power):
        self.__SetHorsepower(power)
        print("Current Horsepower by SetAndGet method: ", self.GetHorsepower())
car=Sol_CCar(200)
car.SetAndGet(500)
