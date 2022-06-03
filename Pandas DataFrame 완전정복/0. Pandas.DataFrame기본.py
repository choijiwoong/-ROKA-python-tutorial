import pandas as pd
import numpy as np

"""
class pandas.DataFrame(data=None, index=None, columns=None, copy=None)는 가변 2차원 배열이다.
data가 dictionary인 경우 Columns의 순서는 dict에 삽입되어있는 순서를 따른다.
index는 행 레이블로 default는 0~이다. columns는 열 레이블로 default는 0~이다.
dtype은 데이터 타입을 강제하고자할 때 사용하며 default는 None이며 자동으로 type이 추론된다.
copy는 DataFrame의 원본의 수정여부(동기화)를 결정하는데, default는 True이며 False로 세팅 시 DataFrame변경시 원본 데이터도 변경된다.
"""
 #1. DataFrame의 copy옵션에 대하여
np.random.seed(0)
arr=np.random.randint(10, size=(2,2))
print("원본데이터: \n", arr)#원본 데이터

df1=pd.DataFrame(arr, copy=False)#수정시 원본 데이터도 같이 수정
df2=pd.DataFrame(arr, copy=True)#수정시 원본 데이터와 독립적으로 수정

arr[0,0]=99#역으로 원본데이터변경시
print('\ndf1:\n', df1)#DataFrame의 값 역시 변경. '

print('\ndf2:\n', df2)#변경X

 #2. DataFrame 예시
data={'A': [1, 2], 'B': [3, 4]}#dictionary
df=pd.DataFrame(data=data)
print(df)#'A'와 'B'가 열정보로, 행은 default 0~으로 설정.
del df

data=np.array([[1,2,], [3,4,]])
df=pd.DataFrame(data=data, index=['row1', 'row2'], columns=['col1', 'col2'])
print(df)#index로 행정보, columns로 열정보
