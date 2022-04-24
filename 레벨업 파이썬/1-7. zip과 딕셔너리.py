#두 개의 리스트를 묶기
name=['merona', 'gugucon']
price=[500, 1000]

z=zip(name, price)#두개의 리스트를 서로 묶어준다.(by tuple)
print(list(z))#[('merona', 500), ('gugucon', 1000)]

for n, p in zip(name, price):
    print(n, p)#와 같이 각각의 원소를 for문으로 쉽게 사용이 가능하다

#딕셔너리의 다양한 생성 방법
아이스크림1={"메로나": 500, "구구콘": 1000}
아이스크림2=dict(메로나=500, 구구콘=1000)
아이스크림3=dict([("메로나", 500), ("구구콘", 1000)])#키와 값을 하나의 튜플로 저장하고 여러 튜플을 리스트로서 전달

#zip과 딕셔너리
icecream=dict(zip(name, price))#하나씩을 가져와 dict으로
print(icecream)

#setdefault메서드(keys, values, items)_키를 추가하면서 초깃값을 설정할 경우 setdefault메서드를 사용할 수 있다.
data={}

ret=data.setdefault('a', 0)#키고 a를 추가하고, value를 0으로. 쉽게 말해 key값이 없으면 value를 리턴한다. default의 기본값은 None이다.
print(ret, data)

ret=data.setdefault('a', 1)#이미 동일한 key가 존재하는 경우 setdefault는 현재 value값을 리턴한다.
print(ret, data)

#딕셔너리 원소 개수
data=["BTC", "BTC", "XRP", "ETH", "ETH", "ETH"]

for k in set(data):#딕셔너리를 set으로 만들어 개수를 세야하는 종목을 get
    count=data.count(k)#딕셔너리에 해당 원소의 개수를 세어
    print(k, count)#출력
