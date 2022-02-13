#[1. two ways for static method(feat. class method)]
class hello:
    num=10

    @staticmethod
    def calc(x):
        return x+10+hello.num#approach class variable by namespace
print(hello.calc(200))


class hi:
    num=10

    @classmethod
    def calc(cls, x):
        return x+10+cls.num#approach class variable by cls
print(hi.calc(200))
#cls means it's class.


#how about cls ininheritance?
class based:
    t="I's owner"

    @classmethod
    def calc(cls):
        return cls.t

class derived(based):
    t="I's worker"

print(derived.calc())#cls mean's derived.


#[2. two ways for abstract class]
class Car:
    def turnning(cls, horsepower):
        raise NotImplementedError#make error when calling method

class Sonata(Car):
    def turnning(self):
        print("turnning finish")
sonata=Sonata()
sonata.turnning()


import abc

class Car2:
    @abc.abstractmethod#make error when calling class_more earily detacting of error of abstract than first way
    def turnning(cls, horsepower):
        pass

class Sonata2(Car2):
    def turnning(self):
        print("turnning finish")
sonata2=Sonata2()
sonata2.turnning()


#[3. meta class]_make customized class
print(type(3))#<class 'int'>

temp=type('temp', (), {})#make temp by using type! each is typle for definition of property & dictionary for definition of method
print(temp)#<class '__main__.temp'>

print(type(int))#<class 'type'>

#example of meta class
class normal_class:
    a=3

    def add(self, m, n):
        return m+n
before=normal_class()
print(before.add(10,20))


after=type('normal_class', (object,), {'a': 3, 'add': lambda m, n: m+n})#tuple means inheritance. so we don't have to inherite object.
print(after.add(10, 20))


#Actual example of metaclass_customized meta class
class Integer_Only(type):
    def __new__(cls, clsname, bases, dct):
        assert type(dct['a']) is int, 'attribute a is not integer'
        return type.__new__(cls, clsname, bases, dct)
class wrong_temp(metaclass=Integer_Only):
    a=3.14
value=wrong_temp()
