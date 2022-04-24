a=3
id(a)#메모리에 할당된 객체의 주소를 a라는 변수가 가리키는 것을 바인딩(Binding)이라고 부른다.

import sys
print(sys.getrefcount(a))#155
#sys.getrefcount()로 객체의 레퍼런스 카운트를 확인할 수 있는데, 변수가 어떤 객체를 바인딩하면 객체의 레퍼런스 카운트 값이 증가한다.
#위의 경우 이미 파이썬 인터프린터가 실행될 때 내부적으로 3을 여러번 참조했기에 큰 값이 나온다.
print(sys.getrefcount(12941284921))#2

print('3의 레퍼런스 카운트: ', sys.getrefcount(a))#155
b=3
c=3
d=4
print('3의 레퍼런스 카운트: ', sys.getrefcount(3))#157


a=[0,1,2]
print('리스트 객체에 관한 참조횟수: ', sys.getrefcount(a))#2
#파이썬에서의 모든 값은 어떤 클래스의 인스턴스이다. 참조될때 refcount가 증가하고, 해제될 때 refcount가 감소한다. 그러다 이 값이 0이 되면 GGC(general)가 해제한다.
