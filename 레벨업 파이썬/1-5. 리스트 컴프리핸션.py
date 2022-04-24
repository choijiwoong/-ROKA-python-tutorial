even_num=[]

for i in range(10):
    if i%2==0:
        even_num.append(i)

print(even_num)#기존


even_num=[i for i in range(10) if i%2==0]#for문부터 읽고 나머지 if를 충족하면 추가된다고 읽으면 된다. 다차원 for을 쓰는 것은 권장하지 않는다.
print(even_num)

odd_num=[i for i in range(10) if i%2==0]
print(odd_num)

pow2_nums=[i*i for i in range(10)]
