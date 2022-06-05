import numpy as np
import pandas as pd

 #1. 반올림_DataFrame.round(decimals=0, args, kwargs)_소수점 기준점, 그 외의 인자는 numpy와의 호환성을 위해 존재. 영향X
col=['col1', 'col2', 'col3']
row=['row1', 'row2', 'row2']
data=np.random.rand(3,3)*100
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

print(df.round(0))
print(df.round(1))
print(df.round(-1))#양의자리수 1자리에서 반올림.

 #2. 합계_DataFrame.sum(axis=None, skipna=None, level=None, numeric_only=None, min_count, kwargs)_NaN존재시 무시여부(defualt=True), 계산에 필요한 숫자의 최소 개수
data=[[1,2,3], [4,5,6], [7,np.NaN, 9]]
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

print(df.sum(axis=0))#열의 합
print(df.sum(axis=1))#행의 합

print(df.sum(axis=0, skipna=False))#위의 계산에서는 NaN이 무시되고 계산되었는데, False지정시 무시하지않고 NaN을 그대로 출력한다.

print(df.sum(axis=1, min_count=3, skipna=True))#skipna(default값 명시했을뿐..)가 True면 NaN이 무시되어야하지만, min_count를 3으로 지정했기에 NaN을 그대로 출력한다. 우선순위가 skipna보다 min_count가 높은 느낌.

 #3. 곱(prod, prodduct)_DataFrame.prod(axis=None, skipna=None, level=None, numeric_only=None, min_count=0, kwargs)
print(df)

print(df.prod(axis=0))#열의 곱
print(df.prod(axis=1))#행의 곱

print(df.prod(axis=0, skipna=False))#NaN포함 시 그대로 반환
print(df.prod(axis=1, min_count=3))#2번과 마찬가지로 skipna default값인 True보다 min_count가 우선. NaN출력.
print(df.product(axis=1, min_count=3))#위와 동일한 연산.

 #4. 절대값_DataFrame.abs()_NaN의 경우 그대로 출력, 복소수의 경우 크기가 반환. a+bj의 크기: sqrt(a*a+b*b)
data=[[-1, 2, -3.5], [4, -5.5, 3+4j], [7, np.NaN, 0]]
df=pd.DataFrame(data=data, index=row, columns=col)
print(df)

print(df.abs())#NaN의 경우 그대로 출력, 복소수의 경우 절댓값으로 크기를 반환. 이는 pandas에서만 그런게 아니라 기초수학에서 복소 평면에 플로팅된 원점에서 복소수 값까지의 벡터길이로 정의해서 그럼.

 #5. 전치_DataFrame.transpose(args, copy=False) & DataFrame.T(args, copy=False)_반환값으로 사본을 반환할지 여부로 args가 여러 dtype으로 이루어진경우 True로 세팅됨
data=[['A', 1, 2], ['B', 3, 4], ['C', 5, 6], ['D', 7, 8]]
df=pd.DataFrame(data=data, index=['row1', 'row2', 'row3', 'row4'], columns=col)
print(df)

print(df.transpose())
print(df.T)#주의할 점은 transpose는 메서드이고 T는 속성이다.

 #6. 순위_DataFrame.rank(axis=0, method='average', numeric_only=None, na_option='keep', ascending=True, pct=False)
#동일 순위일 경우 평균을 반환, na_option으로 NaN을 keep, top, bottom으로 처리법설정가능, pct는 순위를 백분위수형식으로 할지 여부
data = [[5],[5],[pd.NA],[3],[-3.1],[5],[0.4],[6.7],[3]]
row = ['A★','B★','C','D☆','E','F★','G','H','I☆']
df=pd.DataFrame(data=data, index=row, columns=['Value'])#원본데이터의 column을 Value로 지정
print(df)
#(method에 따른 rank. 같은 값일 때의 순위매김 방법을 지정.)
df['average']=df['Value'].rank(method='average')
df['min']=df['Value'].rank(method='min')
df['max']=df['Value'].rank(method='max') 
df['first']=df['Value'].rank(method='first')
df['dense']=df['Value'].rank(method='dense')
print(df)
#(na_option에 따른 결측값 처리)
df['keep']=df['Value'].rank(na_option='keep')#결측값을 그대로 NaN처리
df['top']=df['Value'].rank(na_option='top')#가장 높은 순위 부여
df['bottom']=df['Value'].rank(na_option='bottom')#낮은 순위 부여
df['pct']=df['Value'].rank(pct=True)#결측값을 그대로 두되 순위를 백분위로 표현
print(df)

 #7. 차이_DataFrame.diff(periods=1, axis=0)_열과 열, 행과 행의 차이를 출력(비교할 축과 간격을 지정)
a=[1,2,3,4,5,6,7,8]
b=[1,2,4,8,16,32,64,128]
c=[8,7,6,5,4,3,2,1]
data={'col1': a, 'col2': b, 'col3': c}#이런 식으로 DataFrame을 생성 시 각 dict의 shape는 같아야 한다.
df=pd.DataFrame(data)
print(df)

print(df.diff(axis=0))#행-바로전 행(맨 첫데이터의 경우 NaN처리)
print(df.diff(axis=1))#열-바로전 열

print(df.diff(periods=3))#compare step을 지정. 행-3칸 전 행(고로 첫 3개의 데이터 NaN처리)
print(df.diff(periods=-2))#음수를 periods로 지정하여 역순 비교도 가능

 #8. 차이_ DataFrame.pct_change(periods=1, fill_method='pad', limit=None, freq=None, kwargs)
#객체 내 차이를 현재값과의 백분율로 출력하는 메서드. (다음행-현재행)/현재행. 즉, 현재행 기준으로 다음행과의 차이율
#fill_method: 결측치 대체 값(ffill, bfill), limit: 결측값을 몇개나 대체할지, freq=시계열 API의 증분 설정
a=[1,1,4,4,1,1]
b=[1,2,4,8,16,32]
c=[1,np.NaN,np.NaN,np.NaN,16,64]
data={'col1': a, 'col2': b, 'col3': c}
df=pd.DataFrame(data)
print(df)

print(df.pct_change())#약간 증분느낌인듯 기울기 구할 때 유용할지도..? 어따쓸지 감이 안잡히네
print(df.pct_change(periods=2))
print(df.pct_change(periods=-1))

print(df.pct_change(fill_method='bfill'))#결측값 존재 시 윗값으로 결측값 대체하여 계산
print(df.pct_change(fill_method='ffill'))

print(df.pct_change(limit=2))#결측값 초기 2개까지만 fill_method에 따라 결측값 교체 후 계산하고, 나머지는 NaN유지

 #9. 누적 계산(expanding)_DataFrame.expanding(min_periods=1, center=None, axis=0, method='single')
#연산을 수행할 요소의 최소 갯수(충족X시 NaN출력), center은 호환을 위하므 method는 한 줄씩 수행할지 전체 테이블로 수행할지. (default는 single로 table사용 시 numba라이브러리에서 engine=numba설정필요)
import numba

data={'col1': [1,2,3,4], 'col2': [3,7,5,6]}
idx=['row1', 'row2', 'row3', 'row4']
df=pd.DataFrame(data=data, index=idx)
print(df)

print(df.expanding().sum())#줄을 진행할 때 마다 해당 연산을 누적하는 그 step별 값을 출력

print(df.expanding(min_periods=4).sum())#최소 4개를 누적시킨 다음 결과를 표시

print(df.expanding(axis=1).sum())#열을 기준으로 누적값의 계산

print(df.expanding(method='table').sum(engine='numba'))#대량의 데이터 연산 시 빠른 속도 지원_asm으로 바꿔서 수행(기존에 행별로 수행한 연산을 테이블단위 전체로 수행)

 #10. 기간이동 계산(rolling)_DataFrame.rolling(window, min_period=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')
"""현재 열에 대해 일정 크기의 window를 이용, 추가메서드를 통해 계산. window의 값이 min_periods보다 작으면 NaN출력
center는 bool 레이블을 window의 중간으로 둘지 여부(default=False_우측), win_type은 가중치계산시 사용옵션(triang, agussian..), on은 기준으로 계산할 열 지정, closed는 window가 닫히는 방향, method는 numba여부"""
period=pd.period_range(start='2022-01-13 00:00:00', end='2022-01-13 02:30:00', freq='30T')#30분을 예기하는듯
data={'col1': [1,2,3,4,5,6], 'col2': period}
idx=['row1', 'row2', 'row3', 'row4', 'row5', 'row6']
df=pd.DataFrame(data=data, index=idx)
print(df)#시계열 데이터를 특정 간격으로 하는 array생성 시 유용할 듯(visualization)

print(df.rolling(window=3).sum())#3개의 window간격으로 sum결과를 출력(계산불가부분은 NaN처리)

print(df.rolling(window=3, closed='left').sum())#각 윈도우 별 작업 시 범위를 조정. 3<=x<6
print(df.rolling(window=3, closed='right').sum())#3<x<=6
print(df.rolling(window=3, closed='both').sum())#3<=x<=6
print(df.rolling(window=3, closed='neither').sum())#3<x<6인데 min_periods는 default로 window_size를 따라가기에 3으로 설정된다. 이 값이 계산하고자 하는 값 2개보다 크기에 모두 NaN이 출력된다.
print(df.rolling(window=3, closed='neither', min_periods=2).sum())#min_periods가 window값으로 default세팅된다는 점을 유의하자

print(df.rolling(window=3, center=True).sum())#결과값을 가운데 정렬

print(df.rolling(window=3, win_type='triang').sum())#None시 모든 포인트에 균등한 가중치 가 부여되며, 약간 정규화 느낌일듯. triang은 삼각함수 가중치래(찾아보니 최소값, 최빈값, 최대값으로 이루어진 삼각형모양 분포)
print(df.rolling(window=3, win_type='gaussian').sum(std=3))

print(df.rolling(window='60T', on='col2').sum())#col2기준으로 rolling수행(즉, 이전에는 col1기준으로 period를 계산했는데, 기준을 바꿈으로써 단순히 1~6의 sum계산이 됨)

 #11. 그룹화 계산(groupby)_DataFrame.groupby(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)
#level은 multiIndex에서 레벨 지정, as_index는 그룹화할 내용을 index로 할지의 여부. False 시 기존 인덱스 유지, group_key?, squeeze는 1행,열의 경우 scalar로 출력, observed는 Categorical grouper에 대해 관찰된 값만 표시할지 여부
idx=['A','A','B','B','B','C','C','C','D','D','D','D','E','E','E']
col=['col1','col2','col3']
data=np.random.randint(0,9,(15,3))#(15, 3) shape로 난수생성
df=pd.DataFrame(data=data, index=idx, columns=col).reset_index()#특정 열을 인덱스로 만들 때 사용. 즉, 지금은 idx를 인덱스로 사용하고 있지만, 단순 인덱스를 생성하므로써 idx를 content로 가져옴..같은데
print(df)

print(df.groupby("index"))#추가 메서드 없이 실행 시 DataFrameGroupBy오브젝트가 생성된다.(객체정보출력)

print(df.groupby('index').mean())
print(df.groupby('index').count())
print(df.groupby('index').agg(['sum', 'mean']))#agg는 여러 메서드를 수행할 경우 MultiColumns 형태로 출력시키낟.

def top(df, n=2, col='col1'):#df에 대하여 상위 n개 열을 반환한다.
    return df.sort_values(by=col)[-n:]
print(df.groupby('index').apply(top))#apply를 통해 index기준 group별 연산 수행
print(df.groupby('index', group_keys=False).apply(top))#top연산 시 index가 groupkey로서 사용되여 content와 중복될 경우 group_key사용을 해제하여 일반적인 numeric index를 유지하도록.

df_cat=pd.Categorical(df['index'], categories=['A', 'B', 'C', 'D', 'E', 'F'])#df의 index에 대해 이들로 categorical하여 df_cat생성
print(df_cat)#없던 F가 지정은 되긴함

print(df['col1'].groupby(df_cat).count())#F가 존재하진 않지만 count연산시 포함시켜서 0으로 표시.
print(df['col1'].groupby(df_cat, observed=True).count())#존재하지 않는 F는 계산결과에도 포함시키지 않음. 관찰된 값만 표시

df.loc[6, 'index']=np.NaN#결측값 임의 생성
print(df)
print(df.groupby('index').sum())#default로는 계산 시 NaN은 제외됨
print(df.groupby('index', dropna=False).sum())#결측값 drop을 False화하여 drop하지 않음

idx = [['idx1','idx1','idx2','idx2','idx2'],['row1','row2','row1','row2','row3']]
col = ['col1','col2','col2']
data=np.random.randint(0, 9, (5,3))
df=pd.DataFrame(data=data, index=idx, columns=col).rename_axis(index=['lv0', 'lv1'])#말 그대로 index를 여러개 사용하는 경우
print(df)

print(df.groupby(level=1).sum())#label을 int로 지정하여 lv1이 사용되게 끔
print(df.groupby(['lv1', 'lv0']).sum())#직접 사용할 label을 str로 지정(여러개 지정도 가능)

 #12. 지수가중함수_DataFrame.ewm(com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None, method='single')
#com_질량, span_계산기간, halflife_반감기, alpha_직접입력, adjust_조정계수(나눌 for std)
"""이해가 조금 어려워 추가적으로 자로정리를 함. 지수가중함수는 오래된 데이터에 지수 감쇠를 적용하여 최근 데이터가 더 큰 영향을 끼치도록 가중치를 주는 함수
보통 mean으로 지수가중평균을 사용하며, 그 외에 질량중심, 계산기간, 반감기 등 여러 활성계수의 세팅을 통해 민감도르르 조정할 수 있다.
DataFrame에 assign메서드를 활용하여 추가적인 열 할당이 가능하다.
"""
import matplotlib.pyplot as plt

data = {'val':[1,4,2,3,2,5,13,10,12,14,np.NaN,16,12,20,22]}
df = pd.DataFrame(data).reset_index()#따로 인덱스를 생성 to content
print(df)

df2=df.assign(ewm=df['val'].ewm(alpha=0.3).mean())#df['val']의 지수가중평균 저장
ax=df.plot(kind='bar', x='index', y='val')#df를 출력
ax2=df2.plot(kind='line', x='index', y='ewm', color='red', ax=ax)#df2를 출력
plt.show()

df2=df.assign(ewm_a_low=df['val'].ewm(alpha=0.1).mean())
df3=df.assign(ewm_a_high=df['val'].ewm(alpha=0.7).mean())
ax=df.plot(kind='bar', x='index', y='val')
ax2=df2.plot(kind='line', x='index', y='ewm_a_low', color='red', ax=ax)#지수가중평균 데이터의 활성계수가 높으면 변화에 더 민감하게 나타난다. 
ax3=df3.plot(kind='line', x='index', y='ewm_a_high', color='green', ax=ax)
plt.show()
#a=2/(span+1)
df2=df.assign(span_4=df['val'].ewm(span=4).mean())#spam은 기간을 지정하여 평활계수를 계산하는 인수.. spam이 길면 과거의 데이터 영향이 커진다.
df3=df.assign(span_8=df['val'].ewm(span=8).mean())
ax=df.plot(kind='bar', x='index', y='val')
ax2=df2.plot(kind='line', x='index', y='span_4', color='red', ax=ax)
ax3=df3.plot(kind='line', x='index', y='span_8', color='green', ax=ax)
plt.show()
#a=1/(1+com)
df2=df.assign(com_2=df['val'].ewm(com=2).mean())
df3=df.assign(com_10=df['val'].ewm(com=10).mean())
ax=df.plot(kind='bar', x='index', y='val')
ax2=df2.plot(kind='line', x='index', y='com_2', color='red', ax=ax)
ax3=df3.plot(kind='line', x='index', y='com_10', color='green', ax=ax)
plt.show()
#a=1-e^(-ln(2)/halflife)
df2=df.assign(harf_2=df['val'].ewm(halflife=2).mean())
df3=df.assign(harf_5=df['val'].ewm(halflife=5).mean())
ax=df.plot(kind='bar', x='index', y='val')
ax2=df2.plot(kind='line', x='index', y='harf_2', color='red', ax=ax)
ax3=df3.plot(kind='line', x='index', y='harf_5', color='green', ax=ax)
plt.show()

df2=df.assign(adj_True=df['val'].ewm(alpha=0.2, adjust=True).mean())#balancing, normalization을 위해 조정계수로 나눈ㄷ자.
df3=df.assign(adj_False=df['val'].ewm(alpha=0.2, adjust=False).mean())
ax=df.plot(kind='bar', x='index', y='val')
ax2=df2.plot(kind='line', x='index', y='adj_True', color='red', ax=ax)
ax3=df3.plot(kind='line', x='index', y='adj_False', color='green', ax=ax)
plt.show()

df2=df.assign(ignore_na_True=df['val'].ewm(alpha=0.2, ignore_na=True).mean())#결측값을 무시한다.(참고로 adjust에 따라 가중치가 달라질 수 있다?????)
df3=df.assign(ignore_na_False=df['val'].ewm(alpha=0.3, ignore_na=False).mean())
ax=df.plot(kind='bar', x='index', y='val')
ax2=df2.plot(kind='line', x='index', y='ignore_na_True', color='red', ax=ax)
ax3=df3.plot(kind='line', x='index', y='ignore_na_False', color='green', ax=ax)
plt.show()

import numba
#print(df['val'].ewm(alpha=0.2, method='table').mean(engine='numba')) #호환성 때문인지 ValueError: method='table' not applicable for Series objects. 뜸..
