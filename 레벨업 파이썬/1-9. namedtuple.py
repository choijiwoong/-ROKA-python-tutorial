class Book:
    def __init__(self, title, price):
        self.title=title
        self.price=price

mybook=Book("파이썬을 이용", 27000)
print(mybook.title, mybook.price)

mybook2=("파이썬을 이용", 27000)#tuple
print(mybook2[0], mybook2[1])


from collections import namedtuple#튜플의 성질을 가졌지만, 항목에 이름으로 접근이 가능(index뿐이 아닌 name(key)으로 접근가능한 namedtuple)

Book=namedtuple('Book', ['title', 'price'])#Book클래스에 title, price속성을 추가.
mybook3=Book("홍보좀 작작해", 27000)
print(mybook3.title, mybook3.price)
print(mybook3[0], mybook3[1])#실제 튜플처럼 immutable하며, 인덱싱이 가능하다.
mybook3.price=25000#AttributeError! Immutable.
#title과 price를 키로, 입력하는 "홍보좀 작작해", 2700을 값으로 접근이 가능하다. 혼동을 피하기 위해 namedtuple의 첫인자와 리턴값을 받는 변수의 이름을 통일하다.

#namedtuple unpacking
def print_book(title, price):
    print(title, price)

print_book(*mybook3)
