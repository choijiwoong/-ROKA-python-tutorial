class Person:
    pass
p=Person()#메모리에 2개의 객체가 생성된다.(not 한개!) 하나는 Person()이고 하나는 p이다. 클래스 이름은 클래스 객체를 바인딩한다. 각자 고유한 메모리 공간을 갖는다


#객체 이름 공간에 변수 생성
p.data=3#.는 p가 가리키는 공간을 의미하기에 이 p는 {"data":3}을 저장한다. (이는 클래스 공간이 아닌 객체 공간에 저장된다.)


#클래스 공간에 데이터 저장하기
class Person:
    data=4#모든 객체가 참조하는 공용 정보는 그냥 변수를 정의해주면 된다.

p=Person()#현재 p공간 내에는 data가 없다. Person이름공간에 있을 뿐. 즉, 새로이 생성하지 않는다. 공유정보이기에


#객체는 서로 다른 공간
class Person:
    pass

p1=Person()
p2=Person()
p1.balance=1000
p2.balance=100#각 객체에 저장된다.
