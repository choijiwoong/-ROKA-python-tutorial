import numpy as np
import pandas as pd

 #1. 덧셈(add, radd)
"""
DataFrame.add(other, axis='columns', level=None, fill_value=None)
fill_value를 통해 계산 불가한 값을 대체할 값(None, NaN...)을 지정할 수있다. radd의 경우 피연산자의 순서를 바꾸어 더하는 것이다.
other는 더할 값이며, axis는 더할 레이블을 설정한다. level은 multiindex에서 계산할 index의 레벨이다?
"""
data=[[1, 10, 100], [2, 20, 200], [3, 30, 300]]#(3,3)
col=['col1', 'col2', 'col3']
row=['row1', 'row2', 'row3']
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)#일반적인 데이터프레임의 형성

result=df.add(1)#(3,3)에 scalar값을 add
print(result)
result=df+1#.add()를 호출(오버로딩)
print(result)

data2=[[3], [4], [5]]#(3,1)
df2=pd.DataFrame(data=data2, index=['row1', 'row2', 'row3'], columns=['col1'])
print(df2)#shape가 다른 데이터프레임변수

result=df.add(df2)#(3,3)의 df에 (3,1)의 df2를 add연산
print(result)#계산 불가능한 부분의 값이 NaN으로 대체되었다. 이 값은 add시 fill_value인자로 대체가능하다.

result=df.add(df2, fill_value=0)#add시 계산 불가능한 값을 default value인 NaN에서 0으로 대체
print(result)#(3,3)연산을 위해 df2의 (3,1)외에 부족한 부분의 값을 0으로 대체한 후 add연산을 수행한다.
#즉, 브로드캐스팅이 이루어지는 것이 아니라 빈 값을 fill_value의 값으로 대체한 후 연산을 이어간다.

 #2. 뺄셈(sub, rsub)
print()
"""
DafaFrame.sub(other, axis='columns', level=None, fill_value=None)
뺄 값, 뺄 레이블(0은 행, 1은 열, Series의 경우 Index와 일치시킬 축), multiindex에서 계산할 index의 레벨, 계산전 누락요소 대체값"""
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

result=df.sub(1)
print(result)

result=df-1
print(result)

df2=pd.DataFrame(data=data2, index=['row1', 'row2', 'row3'], columns=['col1'])
print(df2)#다른 shpae의 df

result=df.sub(df2)
print(result)

result=df.sub(df2, fill_value=0)
print(result)

 #3. 곱셈(mul, rmul)_DataFrame.mul(other, axis='columns', level=None, fill_value=None)
print()
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

result=df.mul(2)
print(result)
result=df*2
print(result)

df2=pd.DataFrame(data=data2, index=['row1', 'row2', 'row2'], columns=['col1'])
print(df2)

result=df.mul(df2)
print(result)
result=df.mul(df2, fill_value=0)
print(result)

 #4. 나눗셈(div, rdiv)_DataFrame.div(other, axis='columns', level=None, fill_value=None)
print()
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

result=df.div(2)
print(result)
result=df/2
print(result)

df2=pd.DataFrame(data=data2, index=['row1', 'row2', 'row3'], columns=['col1'])
print(df2)

result=df.div(df2)
print(result)
result=df.div(df2, fill_value=1)#ZeroDivision방지
print(result)

 #5. 나머지(mod, rmod)_DataFrame.mod(other, axis='columns', level=None, fill_value=None) %연산과 동일
print()
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

result=df.mod(7)
print(result)
result=df%7#나머지를 구하는 연산이기에 %와 연산결과가 동일.
print(result)

df2=pd.DataFrame(data=data2, index=['row1', 'row2', 'row3'], columns=['col1'])
print(df2)

result=df.mod(df2)
print(result)
result=df.mod(df2, fill_value=1)
print(result)

 #6. 거듭제곱(pow, rpow)_DataFrame(other, axis='columns', level=None, fill_value=None)
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

result=df.pow(3)
print(result)
result=df**3
print(result)

df2=pd.DataFrame(data=data2, index=['row1', 'row2', 'row3'], columns=['col1'])
print(df2)

result=df.pow(df2)
print(result)
result=df.pow(df2, fill_value=0)
print(result)

 #7. 행렬곱(dot)_DataFrame.dot(other)
col=['col1', 'col2']
row=['row1', 'row2']
data1=[[1,2], [3,4]]
data2=[[5,6], [7,8]]
df1=pd.DataFrame(data=data1)#matrix dot연산 시 columns이나 row가 지정되면 not aligned오류가 뜨네.. 뭐 행렬곱연산이니 값이 중요하며 연산과정에 shape가 바뀌기때문에 그렇겠지. 참고하면 좋을듯
df2=pd.DataFrame(data=data2)
print(df1)
print(df2)

df3=df1.dot(df2)
print(df3)
