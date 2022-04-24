name=['merona', 'gugucon']
price=[500, 1000]

icecream={k:v for k, v in zip(name, price)}
print(icecream)

icecream2={k:v*2 for k, v in zip(name, price)}
print(icecream2)

name=['merona', 'gugucon', 'bibibig']
price=[500, 1000, 600]

icecream={k:v for k,v in zip(name, price) if v<1000}#간단한 조건문
print(icecream)
