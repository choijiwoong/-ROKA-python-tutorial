import pandas as pd

#Series
sr=pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])
print("시리즈 출력: \n", '-'*15, '\n',sr)

print("시리즈의 값: {}".format(sr.values))
print("시리즈의 인덱스: {}".format(sr.index), end='\n\n\n')

#DataFrame
values=[[1,2,3], [4,5,6], [7,8,9]]
index=['one', 'two', 'three']
columns=['A', 'B', 'C']

df=pd.DataFrame(values, index=index, columns=columns)

print("데이터 프레임 출력: \n", '-'*18, '\n', df)

print('데이터프레임의 인덱스: {}'.format(df.index))
print("데이터프레임의 열이름: {}".format(df.columns))
print("데이터프레임의 값: \n", '-'*18, '\n', df.values, end='\n\n\n')

#데이터 프레임의 생성_List, Series, dict, ndarrays, ...etc
data = [
    ['1000', 'Steve', 90.72], 
    ['1001', 'James', 78.09], 
    ['1002', 'Doyeon', 98.43], 
    ['1003', 'Jane', 64.19], 
    ['1004', 'Pilwoong', 81.30],
    ['1005', 'Tony', 99.14],
]
df=pd.DataFrame(data, columns=['학번', '이름', '점수'])
print(df)

data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
}
df=pd.DataFrame(data)
print(df, end='\n\n\n')

#데이터프레임 조회하기_head, tail, []
print(df.head(3))
print(df.tail(3))
print(df['학번'], end='\n\n\n')

#외부 데이터 읽기
try:
    df=pd.read_csv('example.csv')
    print(df)
    print(df.index)
except FileNotFoundError as e:
    print("file is not exist!", end='\n\n\n')
