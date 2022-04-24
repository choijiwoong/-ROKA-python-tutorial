a="hello"
b=["hello", "python"]

print(id(a), id(b))#2787404676208 2787373377792
print(id(a), id(b[0]))#같은 hello를 가리키는 변수의 주소를 출력하면 같다. 1927687175280 1927687175280
#b의 index0이 가리키는 "hello"를 a가 동일하게 가리키고 있기 때문이다.


#파이썬에서는 Immutable 객체로 int, float, str, tuple / Mutable 객체로 list, dict 타입이 있다.
a="python2"
print(id(a))#2493093117360
a="python3"
print(id(a))#2493093117424
#이때의 메모리상의 변화는 "python2"를 a가 바인딩했다가 다른 문자열 객체를 바인딩하며, "python2"의 참조횟수가0이 되고(아무도 참조하지 않기에) GC에 의해 자동 소멸된다.
#문자열 객체는 Immutable객체이기에 기존 객체가 그대로 남아있기 때문이다.

a=["python2", "python3"]
print('\n',id(a))
a.append("python4")#리스트에 원소추가
print(a)
print(id(a), id(a[0]), id(a[1]), id(a[2]))#append이후에도 리스트 객체 시작주소는 변하지 않는다.
